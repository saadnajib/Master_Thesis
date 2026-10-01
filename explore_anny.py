import sys
sys.path.append('/netscratch/najib/multi-hmr/')

# --- MOCK GRAPHICS LIBRARIES FOR LOGIN NODE ---
# This prevents the script from crashing when it searches for missing 3D drivers!
import unittest.mock as mock
sys.modules['pyrender'] = mock.MagicMock()
sys.modules['OpenGL'] = mock.MagicMock()
sys.modules['OpenGL.GL'] = mock.MagicMock()
sys.modules['OpenGL.platform'] = mock.MagicMock()
# ----------------------------------------------

from datasets.annyone import AnnyOne
import torch

print("Loading AnnyOne Dataset...")
dataset = AnnyOne(data_folder='/netscratch/najib/anydataset/', img_size=672)

# 1. Total Samples
print(f"\nTotal Samples in Dataset: {len(dataset)}")

# 2. Get one sample
img, annot = dataset[0]

print("\n--- DATASET COMPOSITION (One Sample) ---")
print(f"- image: Tensor of shape {img.shape}")
for key, value in annot.items():
    if hasattr(value, 'shape'):
        print(f"- {key}: Tensor of shape {value.shape}")
    elif isinstance(value, list):
        print(f"- {key}: List of length {len(value)}")
    else:
        print(f"- {key}: {type(value)} = {value}")

print("\n--- LOOKING FOR JOINT NAMES ---")
# Let's check if the dataset class has the joint names built-in!
if hasattr(dataset, 'joint_names'):
    for i, name in enumerate(dataset.joint_names):
        print(f"Anny Joint [{i}] = {name}")
elif hasattr(dataset, 'bone_names'):
    for i, name in enumerate(dataset.bone_names):
        print(f"Anny Joint [{i}] = {name}")
else:
    print("Could not find joint names directly in the dataset class.")
    print("You may need to check the Anny documentation link Saif sent!")