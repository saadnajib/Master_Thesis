"""
Inspect a Multi-HMR checkpoint and report which model class it actually fits.

WHY: train.py loads pretrained weights with strict=False, which silently drops
every key whose name doesn't match. A checkpoint from the original repo was
trained with the Multi_HMR class (multi_hmr_anny/multi_hmr.py), whose module
names differ from the Model class in model.py that train.py instantiates. If
they don't match, --pretrained appears to work but loads nothing, and you end
up training from random init without noticing.

Usage:
    python inspect_ckpt.py --ckpt /path/to/multiHMR_672_L_anny.pt
    python inspect_ckpt.py --ckpt /path/to/ckpt.pt --compare
"""

import os
# Must be set BEFORE importing model.py/utils, which pull in the renderer.
# Without this, importing on a login node fails with "Unable to load OpenGL library".
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse
import collections
import torch


def get_state_dict(ckpt):
    """Checkpoints vary: {'model_state_dict':...}, {'state_dict':...}, or raw."""
    if isinstance(ckpt, dict):
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], key
        # Maybe the dict IS the state dict (values are tensors)
        vals = list(ckpt.values())
        if vals and all(torch.is_tensor(v) for v in vals[:5]):
            return ckpt, '<root>'
    raise RuntimeError(f"Could not locate a state dict. Top-level keys: "
                       f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")


def prefix_summary(sd, depth=1):
    """Group parameter names by their top-level module prefix."""
    counts = collections.OrderedDict()
    params = collections.OrderedDict()
    for k, v in sd.items():
        pre = '.'.join(k.split('.')[:depth])
        counts[pre] = counts.get(pre, 0) + 1
        params[pre] = params.get(pre, 0) + (v.numel() if torch.is_tensor(v) else 0)
    return counts, params


def compare(sd, model, label):
    """Report how many checkpoint tensors would actually load into `model`."""
    msd = model.state_dict()
    matched, shape_mismatch = [], []
    for k, v in sd.items():
        if k in msd:
            if torch.is_tensor(v) and msd[k].shape == v.shape:
                matched.append(k)
            else:
                shape_mismatch.append((k, tuple(v.shape), tuple(msd[k].shape)))
    missing = [k for k in msd if k not in sd]
    unexpected = [k for k in sd if k not in msd]

    n_ck = len(sd)
    n_md = len(msd)
    pct = 100.0 * len(matched) / max(n_md, 1)
    print(f"\n--- Compatibility with {label} ---")
    print(f"  checkpoint tensors      : {n_ck}")
    print(f"  model tensors           : {n_md}")
    print(f"  MATCHED (name+shape)    : {len(matched)}   ({pct:.1f}% of model)")
    print(f"  name match, shape differ: {len(shape_mismatch)}")
    print(f"  missing (model has, ckpt lacks) : {len(missing)}")
    print(f"  unexpected (ckpt has, model lacks): {len(unexpected)}")

    if shape_mismatch:
        print("  shape mismatches (first 10):")
        for k, a, b in shape_mismatch[:10]:
            print(f"    {k}: ckpt{a} vs model{b}")
    if unexpected:
        print(f"  unexpected examples: {unexpected[:5]}")
    if missing:
        print(f"  missing examples   : {missing[:5]}")

    if pct > 90:
        print("  => VERDICT: this checkpoint fits this class. Safe to load.")
    elif pct > 40:
        print("  => VERDICT: PARTIAL fit. Some weights load, some are random.")
        print("     Decide deliberately whether that partial transfer is what you want.")
    else:
        print("  => VERDICT: DOES NOT FIT. Loading with strict=False would train")
        print("     essentially from random init. Do not use as-is.")
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--compare', action='store_true',
                    help='also instantiate Model and Multi_HMR and report key overlap')
    args = ap.parse_args()

    print(f"Loading {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location='cpu')

    print(f"\n=== TOP-LEVEL ===")
    if isinstance(ckpt, dict):
        for k in ckpt.keys():
            v = ckpt[k]
            kind = type(v).__name__
            extra = f" ({len(v)} entries)" if isinstance(v, dict) else ""
            print(f"  {k}: {kind}{extra}")
    else:
        print(f"  <not a dict>: {type(ckpt)}")

    # Training args saved inside the checkpoint, if any
    if isinstance(ckpt, dict) and 'args' in ckpt:
        print(f"\n=== CHECKPOINT ARGS ===")
        a = ckpt['args']
        a = vars(a) if hasattr(a, '__dict__') else a
        for k in sorted(a.keys()):
            print(f"  {k} = {a[k]}")
    else:
        print("\n=== CHECKPOINT ARGS ===\n  (none saved — likely from the original repo)")

    sd, where = get_state_dict(ckpt)
    total = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    print(f"\n=== STATE DICT (from '{where}') ===")
    print(f"  tensors: {len(sd)}   total params: {total:,}")

    counts, params = prefix_summary(sd, depth=1)
    print(f"\n=== TOP-LEVEL MODULE PREFIXES ===")
    print(f"  {'prefix':<28} {'tensors':>8} {'params':>15}")
    for pre in counts:
        print(f"  {pre:<28} {counts[pre]:>8} {params[pre]:>15,}")

    print(f"\n  Interpretation:")
    print(f"    'encoder.*' / 'decoder.*' / 'mlp_pose.*'  -> Multi_HMR (multi_hmr_anny/multi_hmr.py)")
    print(f"    'backbone.*' / 'x_attention_head.*'       -> Model (model.py, used by train.py)")

    if args.compare:
        kw = {}
        if isinstance(ckpt, dict) and 'args' in ckpt:
            a = ckpt['args']
            kw = dict(vars(a) if hasattr(a, '__dict__') else a)
        kw.setdefault('simple_depth_encoding', 1)
        kw.setdefault('person_center', 'head')
        kw.setdefault('num_betas', 11)
        kw['pretrained_backbone'] = 0   # don't download weights just to compare

        try:
            from model import Model
            m = Model(**{k: v for k, v in kw.items() if k != 'backbone_pretrained'})
            compare(sd, m, "Model (model.py — what train.py builds)")
            del m
        except Exception as e:
            print(f"\n--- Could not instantiate Model: {type(e).__name__}: {e}")

        try:
            from multi_hmr_anny.multi_hmr import Multi_HMR
            m2 = Multi_HMR(**kw)
            compare(sd, m2, "Multi_HMR (multi_hmr_anny/multi_hmr.py — original repo)")
            del m2
        except Exception as e:
            print(f"\n--- Could not instantiate Multi_HMR: {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()