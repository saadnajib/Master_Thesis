"""
Dump the internal key structure of one top-level module in a checkpoint.

No model imports at all, so this runs anywhere (login node included) with no
OpenGL/CUDA requirements. Use it to see how a checkpoint names the weights
inside e.g. 'encoder.*', so you can tell whether they map onto another class's
module (e.g. Model.backbone.* in model.py).

Usage:
    python inspect_prefix.py --ckpt models/multiHMR/multiHMR_672_L_anny.pt --prefix encoder
    python inspect_prefix.py --ckpt <ckpt> --prefix decoder --depth 4
"""

import argparse
import collections
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--prefix', type=str, default='encoder',
                    help="top-level module to expand (e.g. encoder, decoder, mlp_pose)")
    ap.add_argument('--depth', type=int, default=3,
                    help='how many dot-separated levels to group by')
    ap.add_argument('--samples', type=int, default=20,
                    help='how many full key names to print verbatim')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    keys = [k for k in sd.keys() if k.split('.')[0] == args.prefix]
    if not keys:
        tops = sorted({k.split('.')[0] for k in sd.keys()})
        print(f"No keys under '{args.prefix}'. Available top-level modules:\n  {tops}")
        return

    print(f"'{args.prefix}.*' contains {len(keys)} tensors "
          f"({sum(sd[k].numel() for k in keys):,} params)\n")

    print(f"=== FIRST {args.samples} KEYS (verbatim, with shapes) ===")
    for k in keys[:args.samples]:
        print(f"  {k}   {tuple(sd[k].shape)}")

    print(f"\n=== LAST 5 KEYS ===")
    for k in keys[-5:]:
        print(f"  {k}   {tuple(sd[k].shape)}")

    counts = collections.OrderedDict()
    for k in keys:
        grp = '.'.join(k.split('.')[:args.depth])
        counts[grp] = counts.get(grp, 0) + 1
    print(f"\n=== GROUPED AT DEPTH {args.depth} ({len(counts)} groups) ===")
    for g, c in list(counts.items())[:40]:
        print(f"  {g:<50} {c:>4} tensors")
    if len(counts) > 40:
        print(f"  ... and {len(counts) - 40} more groups")


if __name__ == '__main__':
    main()