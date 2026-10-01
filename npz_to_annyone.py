#!/usr/bin/env python3
"""Convert a 3DPW SMPL->Anny fit NPZ into Anny-One-style labels.

The fit NPZ stores Anny rotations and a fitting-frame offset (`transl`).  It
was fitted to SMPL with transl=0.  This adapter restores the original 3DPW
source translation and applies the 3DPW world-to-camera transform:

    root_trans = R @ (fit_transl + source_trans) + t
    Qcam[0]   = R @ Qfit[0]
    Qcam[1:]  = Qfit[1:]

The raw Anny parameterization is root_relative_world: only matrix 0 carries a
translation; all other matrices have zero translation.  This script uses only
pickle and NumPy (no torch or Anny installation required).
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def load_pickle(path: str):
    # 3DPW files are Python-2-era pickles and need latin1 decoding.
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project [...,3] camera-space points to [...,2] pixels."""
    q = np.einsum("ij,...j->...i", K, points)
    return q[..., :2] / q[..., 2:3]


def bbox_from_points(points_cam: np.ndarray, K: np.ndarray,
                     width: int, height: int) -> list[int]:
    uv = project(points_cam, K)
    uv = uv[np.isfinite(uv).all(axis=1)]
    if len(uv) == 0:
        return [0, 0, 0, 0]
    # Anny-One uses [x_min, y_min, x_max, y_max].  Use the 24 mapped fit
    # joints available in the pilot; clamp to the declared image rectangle.
    lo = np.floor(np.min(uv, axis=0)).astype(int)
    hi = np.ceil(np.max(uv, axis=0)).astype(int)
    lo = np.maximum(lo, [0, 0])
    hi = np.minimum(hi, [width - 1, height - 1])
    return [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]


def adapt(seq: dict, fit, person: int, width: int, height: int):
    frames = np.asarray(fit["frames"], dtype=np.int64)
    Qfit = np.asarray(fit["anny_rotmat"], dtype=np.float32)
    shape = np.asarray(fit["anny_shape"], dtype=np.float32)
    ofit = np.asarray(fit["transl"], dtype=np.float32)
    n = len(frames)
    if Qfit.shape != (n, 163, 3, 3):
        raise ValueError(f"expected anny_rotmat [{n},163,3,3], got {Qfit.shape}")
    if shape.shape != (n, 11) or ofit.shape != (n, 3):
        raise ValueError("fit NPZ has unexpected anny_shape/transl shape")
    trans = np.asarray(seq["trans"][person], dtype=np.float32)
    cam = np.asarray(seq["cam_poses"], dtype=np.float32)
    K = np.asarray(seq["cam_intrinsics"], dtype=np.float32)
    if np.any(frames < 0) or np.any(frames >= len(cam)):
        raise IndexError("fit frame index is outside sequence cam_poses")

    R = cam[frames, :3, :3]
    t = cam[frames, :3, 3]
    source_trans = trans[frames]
    root_trans = np.einsum("nij,nj->ni", R, ofit + source_trans) + t
    root_rot = np.einsum("nij,njk->nik", R, Qfit[:, 0])

    # Full Anny-One pose matrices. Child rotations remain in the fit's
    # root-relative-world convention; only the root orientation is camera-
    # rotated. Translation is present only on the root matrix.
    pose = np.tile(np.eye(4, dtype=np.float32), (n, 163, 1, 1))
    pose[:, :, :3, :3] = Qfit
    pose[:, 0, :3, :3] = root_rot
    pose[:, 0, :3, 3] = root_trans

    # The saved fit joints are in fitting coordinates and include ofit.  Apply
    # the same rigid transform to them for a bbox: R @ (j_fit + source_trans)+t.
    # This is only for the 24 mapped joints saved in the NPZ.
    fit_joints = np.asarray(fit["anny_joints"], dtype=np.float32)
    if fit_joints.shape[:2] != (n, 24):
        raise ValueError(f"expected anny_joints [{n},24,3], got {fit_joints.shape}")
    joints_cam = np.einsum("nij,nkj->nki", R, fit_joints) + np.einsum(
        "nij,nj->ni", R, source_trans)[:, None, :] + t[:, None, :]

    return {
        "frames": frames,
        "K": K,
        "root_trans": root_trans.astype(np.float32),
        "root_rotmat": root_rot.astype(np.float32),
        "pose": pose,
        "shape": shape,
        "joints_cam": joints_cam.astype(np.float32),
        "source_trans": source_trans,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq_pkl", required=True, help="original 3DPW sequence pickle")
    ap.add_argument("--fit_npz", required=True, help="SMPL->Anny pilot fit NPZ")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--person", type=int, default=0, help="3DPW person index (default: 0)")
    ap.add_argument("--frame", type=int, default=11,
                    help="sequence frame for the Anny-One pkl (default: 11)")
    ap.add_argument("--image_width", type=int, default=1080,
                    help="image width in pixels (3DPW portrait default: 1080)")
    ap.add_argument("--image_height", type=int, default=1920,
                    help="image height in pixels (3DPW portrait default: 1920)")
    args = ap.parse_args()

    seq = load_pickle(args.seq_pkl)
    fit = np.load(args.fit_npz, allow_pickle=True)
    if args.person < 0 or args.person >= len(seq["trans"]):
        raise IndexError(f"person {args.person} outside 0..{len(seq['trans'])-1}")
    out = adapt(seq, fit, args.person, args.image_width, args.image_height)
    frames = out["frames"]
    where = np.flatnonzero(frames == args.frame)
    if len(where) != 1:
        raise ValueError(f"requested frame {args.frame} is not unique in fit frames")
    i = int(where[0])
    os.makedirs(args.out_dir, exist_ok=True)

    # Camera fields match the uploaded Anny-One sample format.
    K = out["K"]
    human = {
        "name": f"{Path(args.seq_pkl).stem}_person{args.person}_frame{args.frame:05d}",
        "anny_pose": out["pose"][i],
        "anny_shape": out["shape"][i],
        "tight_bbox": bbox_from_points(out["joints_cam"][i], K,
                                        args.image_width, args.image_height),
        "segm_color": int(args.person + 1),
    }
    frame_path = Path(args.out_dir) / f"{Path(args.seq_pkl).stem}_frame_{args.frame:05d}.pkl"
    with open(frame_path, "wb") as f:
        pickle.dump({
            "focal": np.asarray([K[0, 0], K[1, 1]], dtype=np.float32),
            "princpt": np.asarray([K[0, 2], K[1, 2]], dtype=np.float32),
            "humans": [human],
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    npz_path = Path(args.out_dir) / "pilot60_camspace.npz"
    np.savez(npz_path, frames=frames, root_trans=out["root_trans"],
             root_rotmat=out["root_rotmat"], anny_shape=out["shape"],
             source_trans=out["source_trans"],
             focal=np.asarray([K[0, 0], K[1, 1]], dtype=np.float32),
             princpt=np.asarray([K[0, 2], K[1, 2]], dtype=np.float32),
             image_size=np.asarray([args.image_width, args.image_height], dtype=np.int32),
             person=np.asarray(args.person, dtype=np.int32),
             sequence=np.asarray(Path(args.seq_pkl).name))
    print(f"wrote {frame_path}")
    print(f"wrote {npz_path}")
    print(f"frames={frames.tolist()} person={args.person} image_size=(W={args.image_width}, H={args.image_height})")
    print(f"frame {args.frame} bbox={human['tight_bbox']} root_trans={out['root_trans'][i].tolist()}")


if __name__ == "__main__":
    main()
