#!/usr/bin/env python
"""
build_threedpw_anny_labels.py -- STEP 3 of 3: per-image Anny labels for 3DPW.

Reads every <split>/<seq>_p<person>.npz written by the array job, merges the
people of each sequence frame by frame, and writes ONE label file per image in
the Anny-One format the AnnyOne loader uses:

    labels/<split>/<seq>/image_NNNNN.pkl
        focal     [fx, fy]            princpt [cx, cy]
        imagename  absolute path of the 3DPW jpg
        humans    [ {name, anny_pose (163,4,4), anny_shape (11,),
                     tight_bbox [x0,y0,x1,y1], segm_color, joint_err_mm,
                     visible_frac, gender}, ... ]      one entry per person

plus labels/index_<split>.txt (one label path per line) and a summary.

CAMERA CONVENTION (verified earlier against 3DPW's own 2D detections at
8.9 px, see adapter_check.md): the fit lives in SMPL's transl=0 model frame,
3DPW gives the world translation 'trans' and the world->camera 'cam_poses'.
    root_trans  = R @ (fit_transl + trans) + t
    pose[0]     = R @ Q_fit[0]        (root rotation into camera frame)
    pose[1:]    = Q_fit[1:]           (child rotations unchanged)
Only matrix 0 carries a translation, as in Anny-One.

FILTERS (why a frame can be dropped): joint error above --max_err (fit did
not converge), visible fraction below --min_visible (person mostly outside the
image), or a bbox smaller than --min_box pixels. The counts are printed.

Pure NumPy - no torch or Anny needed; runs on a CPU node in a minute.

USAGE
    python build_threedpw_anny_labels.py --fits threedpw_anny_full \\
        --images data/3DPW/imageFiles --out threedpw_anny_full/labels
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np


def load_fit(path):
    d = np.load(path, allow_pickle=True)
    need = ('anny_rotmat', 'anny_shape', 'transl', 'trans_smpl', 'cam_poses', 'cam_intrinsics',
            'frames', 'joint_err_mm', 'tight_bbox', 'visible_frac', 'img_size', 'person', 'seq_name')
    missing = [k for k in need if k not in d.files]
    if missing:
        raise KeyError(f"{path}: missing {missing} - converted with an old smpl_to_anny.py? "
                       f"(v2.2 or later is needed for tight_bbox / visible_frac)")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fits', required=True, help='root with <split>/<seq>_p<person>.npz')
    ap.add_argument('--images', required=True, help='3DPW imageFiles/ root')
    ap.add_argument('--out', required=True, help='output root for labels/')
    ap.add_argument('--splits', default='train,validation,test')
    ap.add_argument('--max_err', type=float, default=80.0, help='drop frames with joint error above (mm)')
    ap.add_argument('--min_visible', type=float, default=0.3, help='drop people with less of the mesh in view')
    ap.add_argument('--min_box', type=int, default=40, help='drop boxes narrower/shorter than this (px)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    grand = dict(images=0, people=0, dropped_err=0, dropped_vis=0, dropped_box=0)
    for split in [s.strip() for s in args.splits.split(',') if s.strip()]:
        fits = sorted(glob.glob(os.path.join(args.fits, split, '*_p*.npz')))
        if not fits:
            print(f"[{split}] no fits found under {os.path.join(args.fits, split)}")
            continue
        # group the per-person fits by sequence
        by_seq = {}
        for f in fits:
            base = os.path.basename(f)[:-4]
            seq, p = base.rsplit('_p', 1)
            by_seq.setdefault(seq, []).append((int(p), f))

        index, stats = [], dict(images=0, people=0, dropped_err=0, dropped_vis=0, dropped_box=0)
        for seq in sorted(by_seq):
            per_frame = {}            # frame id -> list of human dicts
            K = None
            for person, f in sorted(by_seq[seq]):
                d = load_fit(f)
                K = np.asarray(d['cam_intrinsics'], dtype=np.float32)
                gender = str(d['gender']) if 'gender' in d.files else ''
                Q = np.asarray(d['anny_rotmat'], dtype=np.float32)          # [N,163,3,3]
                shape = np.asarray(d['anny_shape'], dtype=np.float32)       # [N,11]
                tr = np.asarray(d['transl'], dtype=np.float32)             # [N,3]
                ts = np.asarray(d['trans_smpl'], dtype=np.float32)         # [N,3]
                cam = np.asarray(d['cam_poses'], dtype=np.float32)         # [N,4,4]
                err = np.asarray(d['joint_err_mm'], dtype=np.float32)
                box = np.asarray(d['tight_bbox'], dtype=np.int32)
                vis = np.asarray(d['visible_frac'], dtype=np.float32)
                for i, fid in enumerate(np.asarray(d['frames']).tolist()):
                    if not np.isfinite(err[i]) or err[i] > args.max_err:
                        stats['dropped_err'] += 1; continue
                    if vis[i] < args.min_visible:
                        stats['dropped_vis'] += 1; continue
                    if (box[i, 2] - box[i, 0]) < args.min_box or (box[i, 3] - box[i, 1]) < args.min_box:
                        stats['dropped_box'] += 1; continue
                    R, t = cam[i, :3, :3], cam[i, :3, 3]
                    pose = np.tile(np.eye(4, dtype=np.float32), (Q.shape[1], 1, 1))
                    pose[:, :3, :3] = Q[i]
                    pose[0, :3, :3] = R @ Q[i, 0]
                    pose[0, :3, 3] = R @ (tr[i] + ts[i]) + t
                    per_frame.setdefault(int(fid), []).append({
                        'name': f"{seq}_p{person}",
                        'anny_pose': pose,
                        'anny_shape': shape[i],
                        'tight_bbox': box[i].tolist(),
                        'segm_color': int(person + 1),
                        'joint_err_mm': float(err[i]),
                        'visible_frac': float(vis[i]),
                        'gender': gender,
                    })
            out_dir = os.path.join(args.out, split, seq)
            os.makedirs(out_dir, exist_ok=True)
            n_img = 0
            for fid in sorted(per_frame):
                img = os.path.join(os.path.abspath(args.images), seq, f"image_{fid:05d}.jpg")
                if not os.path.isfile(img):
                    continue
                label = {
                    'imagename': img,
                    'focal': np.array([K[0, 0], K[1, 1]], dtype=np.float32),
                    'princpt': np.array([K[0, 2], K[1, 2]], dtype=np.float32),
                    'humans': per_frame[fid],
                }
                path = os.path.join(out_dir, f"image_{fid:05d}.pkl")
                with open(path, 'wb') as fh:
                    pickle.dump(label, fh, protocol=pickle.HIGHEST_PROTOCOL)
                index.append(path)
                n_img += 1
                stats['people'] += len(per_frame[fid])
            stats['images'] += n_img
            print(f"[{split}] {seq:<36} {len(by_seq[seq])} person(s)  {n_img:5d} images")

        with open(os.path.join(args.out, f"index_{split}.txt"), 'w') as fh:
            fh.write("\n".join(index) + ("\n" if index else ""))
        print(f"[{split}] {stats['images']} images, {stats['people']} people; dropped "
              f"{stats['dropped_err']} (error>{args.max_err:g} mm), {stats['dropped_vis']} "
              f"(visible<{args.min_visible:g}), {stats['dropped_box']} (box<{args.min_box}px) "
              f"-> {os.path.join(args.out, f'index_{split}.txt')}\n")
        for k in grand:
            grand[k] += stats[k]
    print(f"TOTAL: {grand['images']} label files, {grand['people']} people")
    return 0


if __name__ == '__main__':
    sys.exit(main())
