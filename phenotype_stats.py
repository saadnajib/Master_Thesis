"""
Measure the ground-truth phenotype distribution of the AnnyOne dataset.

WHY: on real photos the shape head predicts proportions~0.8 and age~0.74 for
every person. Whether that is a BIAS (bad) or the DATASET PRIOR (expected)
depends entirely on what AnnyOne's GT phenotypes actually average. This
script answers that directly. It also tells us whether 0.5 is a sensible
"neutral" value for --neutral_phenotypes, or whether the dataset mean is the
better neutral point.

Usage (login node is fine, no GPU needed; ~1-2 min for 2000 samples):
    python phenotype_stats.py --n 2000
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

from datasets.annyone import AnnyOne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_folder', default='/netscratch/najib/anydataset/')
    ap.add_argument('--n', type=int, default=2000, help='samples to measure')
    ap.add_argument('--img_size', type=int, default=672)
    args = ap.parse_args()

    ds = AnnyOne(data_folder=args.data_folder, img_size=args.img_size)
    N = len(ds)
    idx = np.linspace(0, N - 1, min(args.n, N)).astype(int)
    print(f"dataset size {N}; sampling {len(idx)} evenly spaced items")

    labels = None
    rows = []
    for k, i in enumerate(idx):
        try:
            s = ds[i]
            y = s[1] if isinstance(s, tuple) else s
            shp = y.get('anny_shape', None)
            if shp is None:
                if k == 0:
                    print("keys available:", list(y.keys()))
                    print("ERROR: no 'anny_shape' key in sample - check dataset field name")
                    return
                continue
            shp = np.asarray(shp, dtype=np.float32).reshape(-1, 11) if np.asarray(shp).size % 11 == 0 else np.asarray(shp).reshape(-1)
            rows.append(shp.reshape(-1, 11))
        except Exception as e:
            print(f"  sample {i} failed: {e}")
        if k % 500 == 0 and k:
            print(f"  ...{k} samples")

    if not rows:
        print("no shape data collected"); return
    P = np.concatenate(rows, 0)  # [M, 11]
    names = ['gender','age','muscle','weight','height','proportions',
             'cupsize','firmness','african','asian','caucasian']
    print(f"\ncollected {P.shape[0]} people\n")
    print(f"{'phenotype':<12} {'mean':>6} {'std':>6} {'min':>6} {'max':>6}   note")
    for j, n in enumerate(names):
        m, sd, lo, hi = P[:, j].mean(), P[:, j].std(), P[:, j].min(), P[:, j].max()
        note = ""
        if n in ('proportions', 'age'):
            note = f"<- demo predicts ~{0.80 if n=='proportions' else 0.74:.2f} on real photos"
        print(f"{n:<12} {m:>6.3f} {sd:>6.3f} {lo:>6.3f} {hi:>6.3f}   {note}")

    print("\nHOW TO READ THIS:")
    print("  If the GT mean for proportions/age is close to 0.8/0.74, the demo is")
    print("  predicting the dataset PRIOR, not a bias -> set --neutral_phenotypes")
    print("  to those mean values rather than 0.5, or drop the flag.")
    print("  If the GT mean is near 0.5 with wide spread, the real-photo values ARE")
    print("  a bias -> keep neutralizing to 0.5.")


if __name__ == '__main__':
    main()
