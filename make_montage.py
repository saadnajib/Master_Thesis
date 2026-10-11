"""
make_montage.py -- put the demo outputs of several models side by side.

For every image file name that exists in ALL the given output folders, writes
one picture: the models' renders left to right, each with its label on top.
Pure PIL (no torch, no pyrender), runs on the login node.

USAGE (after run_demo_compare.sh):
    python make_montage.py \
        --folders demo_out_anny_distill_vits_v6b_00140 \
                  demo_out_anny_distill_vits_v7_mixed_00130 \
                  demo_out_anny_distill_vits_v6c_00149 \
        --labels  "v6b (synthetic)" "v7 (+real 3DPW)" "v6c (batch 16)" \
        --out     demo_montage

Options:
    --max_height 900    scale every panel to this height (pixels)
    --only name1 name2  restrict to some file names (without folder)
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

EXTS = ('.jpg', '.jpeg', '.png')


def list_images(folder):
    return {f for f in os.listdir(folder)
            if f.lower().endswith(EXTS) and not f.startswith('._')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folders', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_height', type=int, default=900)
    ap.add_argument('--only', nargs='*', default=None)
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(f.rstrip('/')) for f in args.folders]
    if len(labels) != len(args.folders):
        print("ERROR: --labels must have one entry per folder"); sys.exit(1)
    for f in args.folders:
        if not os.path.isdir(f):
            print(f"ERROR: folder not found: {f}"); sys.exit(1)

    sets = [list_images(f) for f in args.folders]
    common = sorted(set.intersection(*sets))
    if args.only:
        common = [c for c in common if c in set(args.only)]
    if not common:
        print("No image name is present in all folders. What each folder contains:")
        for f, s in zip(args.folders, sets):
            print(f"  {f}: {sorted(s)[:8]}{' ...' if len(s) > 8 else ''}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    band = 44
    for name in common:
        panels = []
        for f in args.folders:
            im = Image.open(os.path.join(f, name)).convert('RGB')
            if im.height != args.max_height:
                w = int(round(im.width * args.max_height / im.height))
                im = im.resize((w, args.max_height), Image.LANCZOS)
            panels.append(im)
        W = sum(p.width for p in panels)
        canvas = Image.new('RGB', (W, args.max_height + band), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for p, lab in zip(panels, labels):
            canvas.paste(p, (x, band))
            draw.text((x + 12, 6), lab, fill=(0, 0, 0), font=font)
            x += p.width
        out = os.path.join(args.out, os.path.splitext(name)[0] + '_montage.jpg')
        canvas.save(out, quality=92)
        print(f"  {out}  ({len(panels)} panels, {canvas.width}x{canvas.height})")
    print(f"\n{len(common)} montage(s) -> {args.out}")


if __name__ == '__main__':
    main()
