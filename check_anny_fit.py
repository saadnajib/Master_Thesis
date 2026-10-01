#!/usr/bin/env python
"""
check_anny_fit.py - sanity-check the .npz written by smpl_to_anny.py.

Reads (keys exactly as saved by smpl_to_anny.py):
    anny_rotmat   [F,163,3,3]   anny_shape [F,11]   transl [F,3]
    joint_err_mm  [F]           frames [F]          smpl_idx/anny_idx [M]
  optional (added by the patched smpl_to_anny.py):
    smpl_joints   [F,24,3]  anny_joints [F,24,3]  phenotype_labels [11]

Checks, CPU only:
  1. contents + rotation-matrix validity (det=+1, orthonormal)
  2. joint-error statistics + histogram
  3. phenotype consistency across frames (same person => ~constant)
  4. overlay SMPL vs Anny skeletons per frame (front/side/top) + per-joint
     error ranking + automatic left/right mirror test
     (needs smpl_joints / anny_joints - otherwise tells you to re-run fit)

Usage:
  python check_anny_fit.py /netscratch/najib/multi-hmr/threedpw_anny_fit.npz \
         --out /netscratch/najib/multi-hmr/anny_fit_check
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SMPL_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16,
                17, 18, 19, 20, 21]


def section(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default="anny_fit_check")
    ap.add_argument("--max-frames", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.npz, allow_pickle=True)
    problems = []

    # ---------------------------------------------------------------- 1
    section("1. CONTENTS OF " + args.npz)
    for k in d.files:
        v = d[k]
        print(f"  {k:18s} shape={str(v.shape):16s} dtype={v.dtype}")
    F = d["joint_err_mm"].shape[0]
    frames = d["frames"]
    print(f"\n  {F} fitted frames: {frames.tolist()}")
    if "sequence" in d:
        print(f"  sequence: {d['sequence']}")

    R = d["anny_rotmat"].astype(np.float64)              # [F,163,3,3]
    det = np.linalg.det(R)
    orth = np.abs(R @ np.swapaxes(R, -1, -2) - np.eye(3)).max()
    print(f"  rotmats: det range [{det.min():.4f}, {det.max():.4f}]  "
          f"max |R R^T - I| = {orth:.2e}")
    if np.abs(det - 1).max() > 1e-3 or orth > 1e-3:
        problems.append("rotation matrices are not valid rotations")
    if not np.isfinite(R).all():
        problems.append("NaN/Inf in anny_rotmat")

    # how far from identity is each bone? tells you which bones actually moved
    ang = np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    moved = (ang > 5).sum(1)
    print(f"  bones rotated >5deg per frame: {moved.tolist()}  (of {R.shape[1]})")
    if "bone_labels" in d:
        bl = d["bone_labels"]
        top = np.argsort(-ang.mean(0))[:10]
        print("  most-rotated bones (mean deg): " +
              ", ".join(f"{bl[i]}={ang[:, i].mean():.0f}" for i in top))
        unmapped_moved = [bl[i] for i in np.where(ang.mean(0) > 20)[0]
                          if i not in set(d["anny_idx"].tolist()) and i != 0]
        if unmapped_moved:
            print(f"  NOTE: unmapped bones with >20deg rotation (unconstrained, "
                  f"can produce odd meshes): {unmapped_moved[:15]}")

    # ---------------------------------------------------------------- 2
    section("2. JOINT ERROR (mm)")
    e = d["joint_err_mm"].astype(float)
    print(f"  per frame: {np.round(e, 1).tolist()}")
    print(f"  mean {np.nanmean(e):.1f}  median {np.nanmedian(e):.1f}  "
          f"p90 {np.nanpercentile(e, 90):.1f}  max {np.nanmax(e):.1f}")
    print(f"  <30 good: {(e < 30).sum()}   30-50 ok: {((e >= 30) & (e < 50)).sum()}   "
          f"50-80 review: {((e >= 50) & (e < 80)).sum()}   >=80 REJECT: {(e >= 80).sum()}")
    plt.figure(figsize=(6, 3.5))
    plt.hist(e, bins=max(10, F // 2), color="steelblue")
    for x, c in [(30, "g"), (50, "orange"), (80, "r")]:
        plt.axvline(x, color=c, ls="--")
    plt.xlabel("mean joint error (mm)"); plt.ylabel("frames"); plt.tight_layout()
    p = os.path.join(args.out, "errors_hist.png"); plt.savefig(p, dpi=120); plt.close()
    print(f"  -> {p}")

    # ---------------------------------------------------------------- 3
    section("3. PHENOTYPE CONSISTENCY (same person => should be ~constant)")
    ph = d["anny_shape"].astype(float)                      # [F,11]
    names = ([str(x) for x in d["phenotype_labels"]] if "phenotype_labels" in d
             else [f"p{i}" for i in range(ph.shape[1])])
    print(f"  {'param':12s} {'mean':>6s} {'std':>6s} {'min':>6s} {'max':>6s}  verdict")
    unstable = []
    for i, n in enumerate(names):
        c = ph[:, i]; sd = c.std()
        v = "stable" if sd < 0.05 else ("DRIFTING" if sd < 0.15 else "UNSTABLE")
        if v == "UNSTABLE":
            unstable.append(n)
        print(f"  {n:12s} {c.mean():6.2f} {sd:6.2f} {c.min():6.2f} {c.max():6.2f}  {v}")
    if unstable:
        problems.append(f"phenotypes vary per frame: {unstable} -> fit ONE shape per person")
    plt.figure(figsize=(7, 3.5))
    for i, n in enumerate(names):
        plt.plot(frames, ph[:, i], marker="o", label=n)
    plt.ylim(0, 1); plt.xlabel("3DPW frame"); plt.ylabel("phenotype")
    plt.legend(fontsize=7, ncol=4); plt.tight_layout()
    p = os.path.join(args.out, "phenotypes.png"); plt.savefig(p, dpi=120); plt.close()
    print(f"  -> {p}")

    # ---------------------------------------------------------------- 4
    section("4. SMPL vs ANNY OVERLAY")
    if "smpl_joints" not in d or "anny_joints" not in d:
        print("  smpl_joints / anny_joints not in file.")
        print("  -> replace smpl_to_anny.py with the patched version (saves them)")
        print("     and re-run MODE=fit, then run this checker again.")
    else:
        S = d["smpl_joints"].astype(float); A = d["anny_joints"].astype(float)
        mapped = ~np.isnan(A[0, :, 0])
        pj = np.linalg.norm(S - A, axis=-1) * 1000.0          # [F,24] mm
        print(f"  {mapped.sum()}/24 joints mapped; per-joint mean error (mm):")
        for j in np.argsort(-np.nan_to_num(pj.mean(0))):
            if not mapped[j]:
                continue
            m = pj[:, j].mean()
            flag = "  <-- high" if m > 60 else ""
            print(f"    {SMPL_NAMES[j]:16s} {m:6.1f}  {'#' * int(m / 5)}{flag}")

        # left/right mirror test
        swap = list(range(24))
        for i, n in enumerate(SMPL_NAMES):
            if n.startswith("left_"):
                swap[i] = SMPL_NAMES.index("right_" + n[5:])
            elif n.startswith("right_"):
                swap[i] = SMPL_NAMES.index("left_" + n[6:])
        e_as_is = np.nanmean(pj)
        e_swap = np.nanmean(np.linalg.norm(S - A[:, swap], axis=-1)) * 1000
        print(f"\n  L/R mirror test: as-is {e_as_is:.1f} mm   swapped {e_swap:.1f} mm")
        if e_swap < e_as_is:
            problems.append("left/right appear MIRRORED in the SMPL->Anny mapping")
            print("  !! swapped is lower -> mapping is mirrored")
        else:
            print("  ok")

        # consistency: saved error should match recomputed
        rec = np.nanmean(pj, axis=1)
        if np.abs(rec - e).max() > 1.0:
            problems.append("saved joint_err_mm != recomputed from joints (check units)")
        print(f"  recomputed mean err from joints: {np.round(rec, 1).tolist()} "
              f"(saved: {np.round(e, 1).tolist()})")

        for f in range(min(args.max_frames, F)):
            fid = int(frames[f])
            fig = plt.figure(figsize=(12, 4.5))
            for i, (el, az, ttl) in enumerate([(10, -90, "front"), (10, 0, "side"),
                                               (90, -90, "top")]):
                ax = fig.add_subplot(1, 3, i + 1, projection="3d")
                for j, pa in enumerate(SMPL_PARENTS):
                    if pa < 0:
                        continue
                    ax.plot(*zip(S[f, j], S[f, pa]), c="tab:blue", lw=2)
                    if mapped[j] and mapped[pa]:
                        ax.plot(*zip(A[f, j], A[f, pa]), c="tab:red", lw=2, ls="--")
                ax.scatter(*S[f].T, c="tab:blue", s=12, label="SMPL (GT)")
                ax.scatter(*A[f, mapped].T, c="tab:red", s=12, label="Anny (fit)")
                ax.text(*S[f, SMPL_NAMES.index("left_wrist")], "L", fontsize=10)
                ax.text(*S[f, SMPL_NAMES.index("right_wrist")], "R", fontsize=10)
                ax.view_init(elev=el, azim=az); ax.set_title(ttl)
                c = S[f, 0]; lim = np.nanmax(np.abs(np.concatenate([S[f], A[f]]) - c))
                ax.set_xlim(c[0] - lim, c[0] + lim); ax.set_ylim(c[1] - lim, c[1] + lim)
                ax.set_zlim(c[2] - lim, c[2] + lim)
                if i == 0:
                    ax.legend(loc="upper left", fontsize=8)
            fig.suptitle(f"frame {fid}   mean joint err {e[f]:.1f} mm")
            fig.tight_layout()
            p = os.path.join(args.out, f"overlay_frame_{fid:04d}.png")
            fig.savefig(p, dpi=110); plt.close(fig)
            print(f"  -> {p}")

    # ---------------------------------------------------------------- summary
    section("VERDICT")
    if problems:
        for pr in problems:
            print("  !! " + pr)
    else:
        print("  no structural problems found")
    print(f"\n  outputs in {os.path.abspath(args.out)}/  (scp the PNGs to look at them)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
