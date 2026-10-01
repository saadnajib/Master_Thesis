import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

from datasets.annyone import AnnyOne
from datasets.bedlam import collate_fn
from torch.utils.data import DataLoader

print("Initializing AnnyOne Dataset...")
dataset = AnnyOne(data_folder='/netscratch/najib/anydataset/', img_size=448)

print("Initializing DataLoader...")
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

print("Fetching first batch...")
for i, (img_array, annot) in enumerate(loader):
    print(f"--- BATCH {i} SUCCESS ---")
    print(f"Image Array Shape: {img_array.shape}")
    print(f"Annotation Keys: {annot.keys()}")
    print(f"Camera Matrix (K) Shape: {annot['K'].shape}")
    print(f"Humans Pose Tensor Shape: {annot['anny_pose'].shape}")
    break # We only need to test one batch!
    
print("If you see this, your dataloader is perfect!")