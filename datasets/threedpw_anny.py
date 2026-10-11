"""
datasets/threedpw_anny.py -- real 3DPW images with Anny ground truth.

Reads the label files written by build_threedpw_anny_labels.py:
    <labels_root>/index_<split>.txt          one label path per line
    <labels_root>/<split>/<seq>/image_NNNNN.pkl
        imagename  absolute path of the 3DPW jpg
        focal      [fx, fy]      princpt [cx, cy]      (original image pixels)
        humans     [ {name, anny_pose (163,4,4) camera frame, anny_shape (11,),
                      tight_bbox, segm_color, ...}, ... ]

and returns EXACTLY what datasets/annyone.py returns, so train.py's
prepare_gt / collate / evaluate work unchanged:
    img_array  [img_size, img_size, 3]  normalised, aspect kept, zero-padded
    annot      {'imagename', 'K' (3x3 for the padded square), 'humans'}

K: the image is scaled by s = img_size / max(W, H) and centred in a square.
AnnyOne's formula skips the centring offset, which is harmless there (square
renders, principal point at the centre) but 3DPW images are 1080x1920
portrait, so the offset is applied explicitly here:
    fx' = fx*s,  cx' = cx*s + (img_size - W*s)/2   (same for y)
For 3DPW's centred principal point this equals img_size/2, as it must.

Only the five keys AnnyOne's humans carry are passed on (name, anny_pose,
anny_shape, tight_bbox, segm_color); the extra bookkeeping in the label file
(joint_err_mm, visible_frac, gender) is dropped so the collate sees the same
structure as for AnnyOne.
"""
import glob
import os
import pickle

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from utils import normalize_rgb

HUMAN_KEYS = ('name', 'anny_pose', 'anny_shape', 'tight_bbox', 'segm_color')


class ThreeDPWAnny(Dataset):
    def __init__(self, labels_root, split='train', img_size=672, n=0, stride=1, training=True):
        super().__init__()
        self.name = 'threedpw_anny'
        self.split = split
        self.subsample = stride
        self.img_size = img_size
        self.training = training
        index = os.path.join(labels_root, f"index_{split}.txt")
        if os.path.isfile(index):
            with open(index) as f:
                files = [l.strip() for l in f if l.strip()]
        else:
            files = sorted(glob.glob(os.path.join(labels_root, split, '*', '*.pkl')))
        files = [f for f in files if os.path.isfile(f)]
        if stride > 1:
            files = files[::stride]
        if n > 0 and n < len(files):
            # evenly spaced so a small n still covers every sequence
            idx = np.linspace(0, len(files) - 1, n).astype(int)
            files = [files[i] for i in idx]
        self.pkl_files = files
        print(f"Loaded {len(self.pkl_files)} 3DPW-Anny labels ({split}) from {labels_root}.", flush=True)

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        pkl_path = self.pkl_files[idx]
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        img_path = data['imagename']
        img_pil = Image.open(img_path).convert('RGB')
        W, H = img_pil.size

        # same preprocessing as AnnyOne: contain, then pad to a centred square
        img_pil = ImageOps.contain(img_pil, (self.img_size, self.img_size))
        w_s, h_s = img_pil.size
        img_pil = ImageOps.pad(img_pil, size=(self.img_size, self.img_size))
        img_array = normalize_rgb(np.asarray(img_pil), imagenet_normalization=1)

        s = self.img_size / float(max(W, H))
        ox = (self.img_size - w_s) / 2.0
        oy = (self.img_size - h_s) / 2.0
        fx, fy = float(data['focal'][0]), float(data['focal'][1])
        cx, cy = float(data['princpt'][0]), float(data['princpt'][1])
        K = np.eye(3, dtype=np.float32)
        K[0, 0], K[1, 1] = fx * s, fy * s
        K[0, 2], K[1, 2] = cx * s + ox, cy * s + oy

        humans = []
        for h in data['humans']:
            hh = {k: h[k] for k in HUMAN_KEYS if k in h}
            hh['anny_pose'] = np.asarray(hh['anny_pose'], dtype=np.float32)
            hh['anny_shape'] = np.asarray(hh['anny_shape'], dtype=np.float32)
            humans.append(hh)

        annot = {
            'imagename': os.path.basename(img_path),
            'K': K,
            'humans': humans,
        }
        return img_array, annot


from datasets.bedlam import collate_fn  # noqa: E402,F401  (same collate as AnnyOne)
