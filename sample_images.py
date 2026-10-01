"""
Sample images out of a nested dataset folder into a flat directory for demo.py.

WHY THIS EXISTS
---------------
The THREEDPW dataset class needs sequenceFiles/ (the annotations) to load at
all. If only imageFiles/ is present, that class cannot be used - but demo.py
does not need annotations, just a folder of images. This bypasses the dataset
class entirely.

WHAT YOU GIVE UP: no ground truth, so no PVE / MPJPE. Qualitative renders only.
For metrics you need the full dataset including sequenceFiles.

3DPW is video: consecutive frames in a sequence are near-identical. This samples
ACROSS sequences (a few frames from each) rather than taking the first N frames
of one sequence, so the exported set is actually varied.

USAGE
    python sample_images.py \\
        --src /netscratch/pourafkh/Datasets/3DPW_OCC/imageFiles \\
        --out /netscratch/najib/multi-hmr/threedpw_demo_images \\
        --per_seq 2 --max_total 12
"""

import argparse
import os
import shutil
import sys

EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='root folder to search (searched recursively)')
    ap.add_argument('--out', required=True, help='flat output folder for demo.py --img_folder')
    ap.add_argument('--per_seq', type=int, default=2,
                    help='how many frames to take from each sub-folder (sequence)')
    ap.add_argument('--max_total', type=int, default=12, help='hard cap on exported images')
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"ERROR: source folder does not exist: {args.src}")
        sys.exit(1)

    # Group images by their containing folder = one sequence per folder.
    seqs = {}
    for root, _dirs, files in os.walk(args.src):
        imgs = sorted(f for f in files if f.lower().endswith(EXTS))
        if imgs:
            seqs[root] = imgs

    if not seqs:
        print(f"ERROR: no images found under {args.src}")
        print("Contents of that folder:")
        try:
            for e in sorted(os.listdir(args.src))[:20]:
                print("   ", e)
        except Exception as e:
            print("    (could not list:", e, ")")
        sys.exit(1)

    print(f"Found {len(seqs)} sequence folder(s), "
          f"{sum(len(v) for v in seqs.values())} images total\n")

    os.makedirs(args.out, exist_ok=True)
    n_done = 0
    for seq_dir in sorted(seqs):
        if n_done >= args.max_total:
            break
        imgs = seqs[seq_dir]
        # spread the picks across the sequence rather than taking the first ones
        if args.per_seq >= len(imgs):
            picks = imgs
        else:
            step = len(imgs) / float(args.per_seq + 1)
            picks = [imgs[int(step * (k + 1))] for k in range(args.per_seq)]

        seq_name = os.path.basename(seq_dir.rstrip('/')) or 'seq'
        for f in picks:
            if n_done >= args.max_total:
                break
            src = os.path.join(seq_dir, f)
            dst_name = f"{seq_name}__{f}"
            shutil.copy(src, os.path.join(args.out, dst_name))
            print(f"  {dst_name}")
            n_done += 1

    print(f"\nCopied {n_done} images -> {args.out}")
    print(f"  point demo.py at it with:  --img_folder {args.out}")
    print("\nNOTE: no ground truth was copied, so metrics are not possible on")
    print("      these - qualitative renders only.")


if __name__ == '__main__':
    main()
