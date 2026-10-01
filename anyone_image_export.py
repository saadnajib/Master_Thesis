"""
Export images from the AnnyOne dataset into a plain folder so demo.py can run
on them, AND write a per-image FOV map.

WHY THE FOV MAP MATTERS
-----------------------
Every AnnyOne image is rendered with its own camera. Measured on the holdout
set, the field of view ranged from 51 to 110 degrees across just five images.
demo.py normally applies ONE global --fov to every image, which unprojects the
predicted 2D location + depth into the wrong 3D place: meshes come out at the
wrong distance and scale (in the worst case a single body fills the whole
frame). This script writes annyone_fov.json mapping each exported filename to
its true FOV, which demo.py reads via --fov_json.

Usage:
    python anyone_image_export.py --n 10 --split holdout --out /path/to/imgs
"""

import argparse
import json
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

from datasets.annyone import AnnyOne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_folder', type=str, default='/netscratch/najib/anydataset/')
    ap.add_argument('--out', type=str,
                    default='/netscratch/najib/multi-hmr/annyone_demo_images')
    ap.add_argument('--n', type=int, default=10, help='how many images to export')
    ap.add_argument('--split', type=str, default='holdout',
                    choices=['holdout', 'train'],
                    help="'holdout'=last 100 samples (never trained on), 'train'=start of dataset")
    ap.add_argument('--img_size', type=int, default=672)
    ap.add_argument('--fov_json', type=str, default='',
                    help='where to write the filename->FOV map '
                         '(default: <out>/../annyone_fov.json)')
    args = ap.parse_args()

    ds = AnnyOne(data_folder=args.data_folder, img_size=args.img_size)
    n_total = len(ds)
    print(f"AnnyOne dataset size: {n_total}")

    if args.split == 'holdout':
        # Mirrors train.py: the last --val_anny_n (=100) samples are held out.
        pool = list(range(max(0, n_total - 100), n_total))
    else:
        pool = list(range(min(args.n * 4, n_total)))
    indices = pool[:args.n]

    os.makedirs(args.out, exist_ok=True)
    print(f"Exporting {len(indices)} images from '{args.split}' -> {args.out}\n")

    fov_map = {}
    fovs = []
    for i in indices:
        sample = ds[i]
        y = sample[1] if isinstance(sample, tuple) else sample

        imgname = y['imagename']
        if isinstance(imgname, (list, tuple)):
            imgname = imgname[0]

        src = imgname if os.path.isabs(imgname) and os.path.exists(imgname) \
            else os.path.join(args.data_folder, imgname)
        if not os.path.exists(src):
            print(f"  [{i}] MISSING on disk: {src}")
            continue

        dst_name = f"annyone_{args.split}_{i:06d}{os.path.splitext(src)[1]}"
        dst = os.path.join(args.out, dst_name)
        shutil.copy(src, dst)

        # Recover the true field of view from the intrinsics:
        #   fov = 2 * atan(W / (2 * fx)), with W approximated as 2*cx.
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

    # Write the per-image FOV map for demo.py --fov_json
    out_json = args.fov_json or os.path.join(
        os.path.dirname(os.path.abspath(args.out.rstrip('/'))), 'annyone_fov.json')
    with open(out_json, 'w') as f:
        json.dump(fov_map, f, indent=2)
    print(f"\nWrote per-image FOV map ({len(fov_map)} entries) -> {out_json}")
    print(f"  pass to demo.py as:  --fov_json {out_json}")

    if fovs:
        avg = sum(fovs) / len(fovs)
        print(f"\nMean FOV across exported images: {avg:.1f} deg")
        print(f"  -> pass  --fov {avg:.0f}  to demo.py   (fallback only)")
        spread = max(fovs) - min(fovs)
        print(f"  FOV range: {min(fovs):.1f} - {max(fovs):.1f} deg (spread {spread:.1f})")
        if spread > 5.0:
            print("  NOTE: FOV varies a LOT between images. A single global --fov")
            print("        will mis-place meshes on most of them. USE --fov_json.")


if __name__ == '__main__':
    main()
