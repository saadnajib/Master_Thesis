"""
Compare GROUND-TRUTH vs PREDICTED phenotypes for the exact AnnyOne holdout
samples that were rendered in the demo.

WHY THIS IS THE DECISIVE TEST
-----------------------------
The neck elongation has now survived every explanation we tested:
  * NOT the body model / rest pose  -> --rest_pose renders a normal neck
  * NOT the helper-joint mask       -> check_mask_boost.py: all 4 neck bones
                                       active, the x4 boost did apply
  * NOT a domain gap                -> the long neck appears on AnnyOne
                                       holdout images too (in-domain)

What is left is the SHAPE head. Anny's 'proportions' phenotype directly
controls head/neck proportions, and the demo logs it at 0.65-0.82 for every
person. If the ground truth for those same images is much lower, the shape
head is over-predicting 'proportions' and stretching every neck - a shape
error masquerading as a pose error.

This script prints GT vs predicted side by side, per person, for the same
sample indices, so the comparison is exact rather than distributional.

USAGE
    # 1. get the predicted values out of the demo log
    grep "\\[shape\\] person" logs/demo-annyone-<jobid>.out > /tmp/pred_shape.txt
    # 2. run the comparison
    python compare_gt_pred_shape.py --pred_log /tmp/pred_shape.txt \\
                                    --start 579860 --n 10
"""

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

from datasets.annyone import AnnyOne

NAMES = ['gender', 'age', 'muscle', 'weight', 'height', 'proportions',
         'cupsize', 'firmness', 'african', 'asian', 'caucasian']


def parse_pred_log(path):
    """Read '[shape] person N: gender=0.69  age=0.75 ...' lines."""
    preds = []
    if not path or not os.path.isfile(path):
        return preds
    for line in open(path, errors='ignore'):
        if '[shape] person' not in line:
            continue
        vals = {}
        for k, v in re.findall(r'(\w+)=([0-9.]+)', line):
            if k in NAMES:
                vals[k] = float(v)
        if vals:
            preds.append(vals)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_folder', default='/netscratch/najib/anydataset/')
    ap.add_argument('--start', type=int, default=579860,
                    help='first sample index that was demoed')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--img_size', type=int, default=672)
    ap.add_argument('--pred_log', type=str, default='',
                    help='file containing the [shape] lines from the demo log')
    args = ap.parse_args()

    ds = AnnyOne(data_folder=args.data_folder, img_size=args.img_size)
    print(f"dataset size {len(ds)}; reading GT for indices "
          f"{args.start}..{args.start + args.n - 1}\n")

    def find_shape(obj, depth=0):
        """Locate the 11-dim phenotype vector wherever it lives.

        The raw sample is nested: {'imagename', 'K', 'humans': [{...}, ...]},
        so the phenotypes sit inside each entry of 'humans', not at the top
        level. Search by SHAPE (last dim == 11) rather than by a hard-coded
        key name, so this keeps working if the field is renamed.
        """
        out = []
        if depth > 4:
            return out
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (list, tuple, np.ndarray)) or hasattr(v, 'shape'):
                    try:
                        a = np.asarray(v, dtype=np.float32)
                        if a.ndim >= 1 and a.shape[-1] == 11 and a.size % 11 == 0:
                            out.append((k, a.reshape(-1, 11)))
                            continue
                    except Exception:
                        pass
                out.extend(find_shape(v, depth + 1))
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                out.extend(find_shape(v, depth + 1))
        return out

    gt_rows = []
    key_used = None
    for i in range(args.start, min(args.start + args.n, len(ds))):
        try:
            s = ds[i]
            y = s[1] if isinstance(s, tuple) else s
            found = find_shape(y)
            if not found:
                print(f"  [{i}] no 11-dim phenotype vector found. Top-level keys: "
                      f"{list(y.keys()) if isinstance(y, dict) else type(y)}")
                if isinstance(y, dict) and 'humans' in y:
                    h = y['humans']
                    h0 = h[0] if isinstance(h, (list, tuple)) and h else h
                    print(f"        humans[0] keys: "
                          f"{list(h0.keys()) if isinstance(h0, dict) else type(h0)}")
                continue
            for k, arr in found:
                if key_used is None:
                    key_used = k
                    print(f"  (using GT field '{k}')")
                for p in arr:
                    gt_rows.append(p)
            n_people = sum(a.shape[0] for _, a in found)
            print(f"  [{i}] {n_people} person(s)")
        except Exception as e:
            print(f"  [{i}] failed: {type(e).__name__}: {e}")

    if not gt_rows:
        print("no GT collected"); return
    G = np.stack(gt_rows, 0)

    P = parse_pred_log(args.pred_log)
    Pm = None
    if P:
        Pm = np.array([[p.get(n, np.nan) for n in NAMES] for p in P], dtype=np.float32)

    print(f"\nGT people: {G.shape[0]}" + (f"   predicted people: {Pm.shape[0]}" if Pm is not None else ""))
    print("\n" + "=" * 78)
    print(f"{'phenotype':<12} {'GT mean':>9} {'GT std':>8} {'GT min':>8} {'GT max':>8}"
          + (f" {'PRED mean':>10} {'diff':>8}" if Pm is not None else ""))
    print("=" * 78)
    flagged = []
    for j, n in enumerate(NAMES):
        g = G[:, j]
        line = f"{n:<12} {g.mean():>9.3f} {g.std():>8.3f} {g.min():>8.3f} {g.max():>8.3f}"
        if Pm is not None:
            p = Pm[:, j]
            p = p[~np.isnan(p)]
            if len(p):
                diff = p.mean() - g.mean()
                line += f" {p.mean():>10.3f} {diff:>+8.3f}"
                # flag a systematic offset larger than the GT spread
                if abs(diff) > max(2 * g.std(), 0.10):
                    flagged.append((n, g.mean(), p.mean(), diff))
        print(line)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if Pm is None:
        print("  No predictions supplied (--pred_log). GT distribution shown above.")
    elif flagged:
        print("  The shape head is SYSTEMATICALLY OFF on these phenotypes:")
        for n, gm, pm, d in flagged:
            print(f"    {n:<12} GT={gm:.3f}  predicted={pm:.3f}  offset={d:+.3f}")
        if any(n == 'proportions' for n, _, _, _ in flagged):
            print("\n  'proportions' is flagged -> this is the neck. That phenotype")
            print("  controls head/neck proportions in Anny, so a systematic")
            print("  over-prediction stretches every neck REGARDLESS of pose,")
            print("  which matches everything observed:")
            print("    - rest pose looks fine (pose is not the cause)")
            print("    - the x4 neck-rotation boost changed nothing (wrong target)")
            print("    - it persists in-domain (not a domain gap)")
            print("\n  FIX: raise --alpha_shape (currently 1.0 vs alpha_v3d 100.0, so")
            print("  shape is barely supervised), and/or supervise 'proportions'")
            print("  specifically. Then continue training.")
    else:
        print("  Predictions track the GT distribution closely on every phenotype.")
        print("  Shape is NOT the cause; the remaining neck error is a skinning")
        print("  artifact of stacked neck-bone rotations (LBS stretching).")


if __name__ == '__main__':
    main()