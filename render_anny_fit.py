"""
render_anny_fit.py -- overlay fitted Anny meshes on the original 3DPW frames.

WHY
---
The joint error (~34 mm) says the SKELETON matches. It says nothing about
whether the fitted PHENOTYPES describe the right person - e.g. gender=0.02 and
asian=0.76 could be true, or could be the optimiser using ethnicity/gender
morphs to absorb skeletal proportions. The only way to know is to look. This is
also the supervisor's "qualitative visual check" item.

THE FRAME-INDEX BUG (job 3454323) - FIXED HERE
----------------------------------------------
v1 used d['img_frame_ids'] as the image number. In 3DPW that array is NOT
"image number of pose frame t". It is the index into the *60 Hz* pose arrays
(poses_60Hz / trans_60Hz) for each image, i.e. 0, 2, 4, ... roughly 2*t. So
pose frame 25 was drawn on image 50, pose frame 575 on image 1151, and so on:
the person had walked somewhere else by then, which is exactly why the meshes
floated NEXT TO the people with the wrong pose. (Tell-tale in the log: fitted
frames were 0, 25, 50, ... but "rendered" frames were 0, 50, 100, ... and the
renderer asked for image_01401.jpg in a 1387-frame sequence.)

3DPW's 30 Hz 'poses', 'trans', 'cam_poses' and 'campose_valid' are all indexed
by the IMAGE number, so the image for fitted pose frame t is image_{t:05d}.jpg.
That is what this version uses (d['frames']).

HOW THE MESH GETS BACK ONTO THE PHOTO
-------------------------------------
The fit matched SMPL joints computed with transl=0, so the Anny mesh lives in
SMPL's model frame (world orientation, no world translation). 3DPW gives, per
frame, the SMPL world translation 'trans', the world->camera transform
'cam_poses', and the camera 'cam_intrinsics':
    world  = anny_verts + trans[t]
    cam    = R @ world + T          (from cam_poses[t])
    pixels = K @ cam
smpl_to_anny.py stores all of these in the .npz, so this script needs only the
.npz, the image folder and the Anny model.

JOINT OVERLAY (new)
-------------------
If the .npz contains 'target_joints' and 'anny_joints' (written by the v2
smpl_to_anny.py), the SMPL ground-truth joints are drawn as RED dots and the
fitted Anny joints as GREEN dots, joined by a yellow line. This separates the
two questions "is the camera/frame plumbing right?" (red dots on the person)
from "is the fit right?" (green dots on red dots). Old .npz files without those
keys render the mesh only.

USAGE
    python render_anny_fit.py --npz threedpw_anny/<seq>.npz \\
        --img_dir data/3DPW/imageFiles/<seq> --out threedpw_anny/renders/<seq>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import torch


def anny_faces(m):
    """Anny is a quad mesh; renderers want triangles."""
    for attr in ('faces', 'faces_tensor', 'triangles', 'quads'):
        f = getattr(m, attr, None)
        if f is not None:
            f = np.asarray(f.detach().cpu() if torch.is_tensor(f) else f)
            break
    else:
        raise AttributeError("could not find a faces attribute on the Anny model")
    if f.shape[1] == 4:                       # quads -> two triangles each
        f = np.concatenate([f[:, [0, 1, 2]], f[:, [0, 2, 3]]], axis=0)
    return f.astype(np.int64)


def render_overlay(img, verts_cam, faces, K, color=(0.3, 0.6, 1.0), alpha=0.7):
    """Rasterise the mesh with pyrender using the real intrinsics."""
    import pyrender, trimesh
    H, W = img.shape[:2]
    mesh = trimesh.Trimesh(verts_cam, faces, process=False)
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=(*color, 1.0),
                                             metallicFactor=0.1, roughnessFactor=0.8)
    pm = pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=True)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4, 0.4, 0.4])
    scene.add(pm)
    cam = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                                    znear=0.05, zfar=50.0)
    # pyrender looks down -Z; OpenCV-style cameras look down +Z -> flip
    pose = np.eye(4); pose[1, 1] = -1; pose[2, 2] = -1
    scene.add(cam, pose=pose)
    light = pyrender.DirectionalLight(intensity=3.0)
    scene.add(light, pose=pose)
    r = pyrender.OffscreenRenderer(W, H)
    rgba, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    mask = rgba[..., 3:4] / 255.0
    out = img * (1 - alpha * mask) + rgba[..., :3] * (alpha * mask)
    return out.astype(np.uint8)


def to_camera(pts_world, P):
    """world -> camera with a 4x4 world->camera matrix (3DPW cam_poses)."""
    return pts_world @ P[:3, :3].T + P[:3, 3]


def project(pts_cam, K):
    """OpenCV pinhole projection, [N,3] camera points -> [N,2] pixels."""
    z = np.clip(pts_cam[:, 2:3], 1e-6, None)
    uv = (pts_cam / z) @ K.T
    return uv[:, :2]


def draw_joints(img_u8, j_gt_px, j_fit_px, radius=5):
    """Red = SMPL ground-truth joints, green = fitted Anny joints, yellow = residual."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img_u8)
    dr = ImageDraw.Draw(im)
    r = radius
    for a, b in zip(j_gt_px, j_fit_px):
        if np.isfinite(a).all() and np.isfinite(b).all():
            dr.line([tuple(a.tolist()), tuple(b.tolist())], fill=(255, 230, 0), width=2)
    for p in j_gt_px:
        if np.isfinite(p).all():
            dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(255, 40, 40), outline=(0, 0, 0))
    for p in j_fit_px:
        if np.isfinite(p).all():
            dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(40, 255, 40), outline=(0, 0, 0))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--img_dir', required=True, help='3DPW imageFiles/<seq_name>')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max_renders', type=int, default=0,
                    help='render only the first N fitted frames (0 = all)')
    ap.add_argument('--no_joints', action='store_true',
                    help='skip the red/green joint overlay even if the .npz has joints')
    args = ap.parse_args()

    from PIL import Image
    from utils import rotation_to_homogeneous
    import anny

    device = args.device if torch.cuda.is_available() else 'cpu'
    d = np.load(args.npz, allow_pickle=True)
    seq = str(d['seq_name'])
    n = len(d['frames'])
    print(f"{seq}: {n} fitted frames, mean joint err {d['joint_err_mm'].mean():.1f} mm")
    if 'shared_shape' in d.files and np.any(d['shared_shape']):
        print("shared body:", np.round(d['shared_shape'], 2))
    has_joints = (not args.no_joints) and ('target_joints' in d.files) and ('anny_joints' in d.files)
    if has_joints:
        print("joint overlay: red = SMPL ground truth, green = fitted Anny, yellow = residual")
    else:
        print("no joint data in this .npz (old smpl_to_anny.py) - rendering the mesh only")

    m = anny.create_fullbody_model(remove_unattached_vertices=False,
                                   all_phenotypes=True).to(dtype=torch.float32)
    m.shape_keys = [k for k in m.phenotype_labels if k != 'race']
    m.shape_keys.extend(m.phenotype_labels[-3:])
    m.set_skinning_method('lbs'); m.name = 'anny'
    m = m.to(device)
    labels = list(m.phenotype_labels)
    faces = anny_faces(m)
    K = d['cam_intrinsics'].astype(np.float64)

    os.makedirs(args.out, exist_ok=True)
    ok = 0
    n_render = n if args.max_renders <= 0 else min(n, args.max_renders)
    for i in range(n_render):
        # The fitted pose frame index IS the image number (see docstring).
        fid = int(d['frames'][i])
        img_path = os.path.join(args.img_dir, f"image_{fid:05d}.jpg")
        if not os.path.isfile(img_path):
            print(f"  [{i}] missing image {img_path}"); continue
        img = np.asarray(Image.open(img_path).convert('RGB')).astype(np.float32)

        R = torch.as_tensor(d['anny_rotmat'][i], dtype=torch.float32, device=device).unsqueeze(0)
        shp = torch.as_tensor(d['anny_shape'][i], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            out = m(pose_parameters=rotation_to_homogeneous(R),
                    phenotype_kwargs={k: shp[:, j] for j, k in enumerate(labels)})
        v = out['vertices'][0].cpu().numpy() + d['transl'][i]      # model frame
        v_world = v + d['trans_smpl'][i]                              # + SMPL world transl
        P = d['cam_poses'][i]
        v_cam = to_camera(v_world, P)                                 # world -> camera

        if v_cam[:, 2].mean() <= 0:
            print(f"  [{i}] frame {fid}: mesh behind camera (mean z={v_cam[:,2].mean():.2f}) "
                  f"- check cam_poses convention"); continue
        try:
            ov = render_overlay(img, v_cam, faces, K)
        except Exception as e:
            print(f"  [{i}] render failed: {type(e).__name__}: {e}"); continue

        note = ""
        if has_joints:
            jg_cam = to_camera(d['target_joints'][i] + d['trans_smpl'][i], P)
            jf_cam = to_camera(d['anny_joints'][i] + d['trans_smpl'][i], P)
            jg_px, jf_px = project(jg_cam, K), project(jf_cam, K)
            ov = draw_joints(ov, jg_px, jf_px)
            pel = jg_px[0]
            note = f", GT pelvis at pixel ({pel[0]:.0f},{pel[1]:.0f}) of {img.shape[1]}x{img.shape[0]}"

        side = np.concatenate([img.astype(np.uint8), ov], axis=1)
        Image.fromarray(side).save(os.path.join(args.out, f"{seq}_{fid:05d}.jpg"), quality=90)
        ok += 1
        print(f"  [{i}] frame {fid}: err {d['joint_err_mm'][i]:.1f} mm -> rendered{note}")
    print(f"\nrendered {ok}/{n_render} -> {args.out}")


if __name__ == '__main__':
    main()
