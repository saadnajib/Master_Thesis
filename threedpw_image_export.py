"""
Export images from the 3DPW dataset (the licensed copy already on the cluster)
into a plain folder so demo.py can run on them, and write a per-image FOV map.

WHY NOT DOWNLOAD FROM THE INTERNET
----------------------------------
3DPW is distributed under a licence that requires registering and accepting
terms at https://virtualhumans.mpi-inf.mpg.de/3DPW/ . Scraping images from the
web would sidestep that. It is also unnecessary: train.py already imports
datasets.threedpw.THREEDPW and lists THREEDPW in --val_data, so a licensed copy
is already available locally. This script reads that copy through the same
dataset class the training code uses.

WHY THE FOV MAP
---------------
demo.py applies ONE global --fov to every image unless given a per-image map.
3DPW is filmed with a moving hand-held camera and its intrinsics vary between
sequences; a single global value mis-places the meshes in depth and scale
(this exact problem produced "giant blob" renders on AnnyOne). This writes
threedpw_fov.json for demo.py --fov_json.

USAGE
    python threedpw_image_export.py --n 10 --split test --out /path/to/imgs
"""

import argparse
import json
import math
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

from datasets.threedpw import THREEDPW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str,
                    default='/netscratch/najib/multi-hmr/threedpw_demo_images')
    ap.add_argument('--n', type=int, default=10, help='how many images to export')
    ap.add_argument('--split', type=str, default='test',
                    help="3DPW split: usually 'test', 'validation' or 'train'")
    ap.add_argument('--img_size', type=int, default=672)
    ap.add_argument('--stride', type=int, default=0,
                    help='take every Nth sample instead of the first N. 3DPW is video, '
                         'so consecutive frames look almost identical - a stride gives '
                         'more varied images. 0 = auto (spread evenly over the split).')
    ap.add_argument('--fov_json', type=str, default='',
                    help='where to write the filename->FOV map '
                         '(default: <out>/../threedpw_fov.json)')
    args = ap.parse_args()

    # The constructor signature is not guaranteed, and the split names differ
    # between forks ('test' vs 'validation' vs 'val'). Report what the class
    # actually expects instead of failing with a bare TypeError.
    import inspect
    try:
        sig = inspect.signature(THREEDPW.__init__)
        print(f"THREEDPW.__init__ accepts: {list(sig.parameters.keys())[1:]}\n")
    except Exception:
        sig = None

    kw_full = dict(split=args.split, training=0, img_size=args.img_size,
                   subsample=1, n=-1)
    if sig is not None:
        kw_full = {k: v for k, v in kw_full.items() if k in sig.parameters}

    ds = None
    last_err = None
    for split_try in [args.split, 'test', 'validation', 'val', 'train']:
        kw = dict(kw_full)
        if 'split' in kw:
            kw['split'] = split_try
        try:
            ds = THREEDPW(**kw)
            if split_try != args.split:
                print(f"NOTE: split '{args.split}' failed; using '{split_try}' instead.")
                args.split = split_try
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"  split '{split_try}' -> {last_err}")

    if ds is None:
        print(f"\nERROR: could not open 3DPW with any split name.")
        print(f"Last error: {last_err}")
        print("\nThings to check:")
        print("  1. Is 3DPW actually present on this cluster? Look inside")
        print("     datasets/threedpw.py for the hard-coded data path and check it exists.")
        print("  2. What split names does it accept? See --val_split in your")
        print("     training scripts, or the split handling in datasets/threedpw.py.")
        sys.exit(1)

    n_total = len(ds)
    print(f"3DPW '{args.split}' size: {n_total}")

    # 3DPW is video: neighbouring frames are near-duplicates. Spread the
    # samples across the whole split so the exported images are varied.
    if args.stride > 0:
        indices = list(range(0, min(args.n * args.stride, n_total), args.stride))[:args.n]
    else:
        indices = np.linspace(0, n_total - 1, min(args.n, n_total)).astype(int).tolist()

    os.makedirs(args.out, exist_ok=True)
    print(f"Exporting {len(indices)} images -> {args.out}\n")

    fov_map, fovs = {}, []
    for i in indices:
        try:
            s = ds[i]
            y = s[1] if isinstance(s, tuple) else s
        except Exception as e:
            print(f"  [{i}] failed to load: {type(e).__name__}: {e}")
            continue

        imgname = y.get('imagename', None)
        if imgname is None:
            print(f"  [{i}] no 'imagename' field; keys: {list(y.keys())}")
            continue
        if isinstance(imgname, (list, tuple)):
            imgname = imgname[0]

        if not os.path.exists(imgname):
            print(f"  [{i}] MISSING on disk: {imgname}")
            continue

        dst_name = f"threedpw_{args.split}_{i:06d}{os.path.splitext(imgname)[1]}"
        dst = os.path.join(args.out, dst_name)
        shutil.copy(imgname, dst)

        # Recover the true field of view from the intrinsics.
        try:
            K = y['K']
            K = K[0] if getattr(K, 'ndim', 2) == 3 else K
            fx = float(K[0][0])
            cx = float(K[0][2])
            fov = math.degrees(2 * math.atan((2 * cx) / (2 * fx)))
            fovs.append(fov)
            fov_map[dst_name] = round(fov, 3)
            print(f"  [{i}] {dst_name}   fx={fx:.1f}  fov={fov:.1f} deg")
        except Exception as e:
            print(f"  [{i}] {dst_name}   (could not read K: {e})")

    if not fov_map:
        print("\nNo images exported - nothing to demo.")
        sys.exit(1)

    out_json = args.fov_json or os.path.join(
        os.path.dirname(os.path.abspath(args.out.rstrip('/'))), 'threedpw_fov.json')
    with open(out_json, 'w') as f:
        json.dump(fov_map, f, indent=2)
    print(f"\nWrote per-image FOV map ({len(fov_map)} entries) -> {out_json}")
    print(f"  pass to demo.py as:  --fov_json {out_json}")

    if fovs:
        avg = sum(fovs) / len(fovs)
        print(f"\nMean FOV: {avg:.1f} deg   range {min(fovs):.1f} - {max(fovs):.1f}")
        print(f"  -> fallback:  --fov {avg:.0f}")


if __name__ == '__main__':
    main()