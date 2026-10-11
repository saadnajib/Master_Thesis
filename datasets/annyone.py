import os
import glob
import pickle
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from utils import normalize_rgb 

class AnnyOne(Dataset):
    def __init__(self, data_folder, split='training', img_size=448, training=True):
        super().__init__()
        self.name = 'annyone'
        self.split = split
        self.data_folder = data_folder
        self.img_size = img_size
        self.training = training
        
        # Grab all the .pkl annotation files
        self.pkl_files = glob.glob(os.path.join(data_folder, '*.pkl'))
        self.pkl_files.sort()
        print(f"Loaded {len(self.pkl_files)} annotations from Anny-One.")

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        pkl_path = self.pkl_files[idx]
        
        # Determine image path (could be .jpg or .png based on your earlier 'ls' output)
        img_path = pkl_path.replace('.pkl', '.jpg')
        if not os.path.exists(img_path):
            img_path = pkl_path.replace('.pkl', '_seg.png') # adjust if standard images are png
            
        img_pil = Image.open(img_path).convert('RGB')
        real_width, real_height = img_pil.size
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        # 1. Image Preprocessing (Padding to square to match BEDLAM)
        img_pil = ImageOps.contain(img_pil, (self.img_size, self.img_size))
        img_pil = ImageOps.pad(img_pil, size=(self.img_size, self.img_size))
        
        img_array = np.asarray(img_pil)
        img_array = normalize_rgb(img_array, imagenet_normalization=1)

        # 2. Camera Matrix (K) Calculation (Adapted from BEDLAM)
        K = np.eye(3, dtype=np.float32)
        focal = data['focal']
        princpt = data['princpt']
        
        princpt_width = princpt[0] / real_width
        princpt_height = princpt[1] / real_height
        K[[0,1],[-1,-1]] = self.img_size * np.asarray([princpt_width, princpt_height])
        K[[0,1],[0,1]] = focal / (max([real_width, real_height]) / self.img_size)

        # 3. Construct Annotations Dictionary
        annot = {
            'imagename': os.path.basename(img_path),
            'K': K,
            'humans': data['humans'] # The collate_fn will dynamically handle anny_pose/anny_shape!
        }

        return img_array, annot

# Mimic the collate_fn from bedlam.py so the trainer can import it from here if needed
from datasets.bedlam import collate_fn