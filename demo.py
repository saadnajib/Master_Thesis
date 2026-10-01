import pyrender
import os
import sys
from argparse import ArgumentParser
import random
import pickle as pkl
import numpy as np
from PIL import Image, ImageOps

# Detector-guided crops can legitimately exceed PIL's decompression-bomb
# threshold: a person filling a large photo needs a context crop several times
# the image size (mostly black padding), which is harmless but trips the guard.
# These are our own local files, so raise the limit rather than crash.
Image.MAX_IMAGE_PIXELS = None
import torch
from tqdm import tqdm
import time
import math

from utils import normalize_rgb, render_meshes, get_focalLength_from_fieldOfView, demo_color as color, print_distance_on_image, render_side_views, create_scene, MEAN_PARAMS, CACHE_DIR_MULTIHMR, SMPLX_DIR
from model import Model
from multi_hmr_anny.multi_hmr import Multi_HMR as ModelAnny
from pathlib import Path

torch.cuda.empty_cache()

np.random.seed(seed=0)
random.seed(0)
import glob


import pyrender
import trimesh
from PIL import Image

def render_mesh_overlay(img_pil, verts, faces, K):
    img = np.asarray(img_pil).astype(np.uint8)
    h, w = img.shape[:2]

    mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh = pyrender.Mesh.from_trimesh(mesh_tm, smooth=False)

    scene = pyrender.Scene(
        bg_color=[0, 0, 0, 0],
        ambient_light=[0.2, 0.2, 0.2])
    scene.add(mesh)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.05, zfar=10000.0)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()

    alpha = (color[..., 3:4] / 255.0).astype(np.float32)
    rgb = color[..., :3].astype(np.float32)
    out = (rgb * alpha + img.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return Image.fromarray(out)

def open_image(img_path, img_size, device=torch.device('cuda')):
    """ Open image at path, resize and pad """

    # Open and reshape
    img_pil = Image.open(img_path).convert('RGB')
    aspect_ratio = img_pil.width / img_pil.height
    
    # keep the original image with padding for visualisation
    img_pil_full = img_pil.copy()
    # img_pil_full = ImageOps.pad(img_pil_full.copy(), size=(max(img_pil_full.size),max(img_pil_full.size)), color=(255, 255, 255))

    # Resize while keeping aspect ratio
    img_pil = ImageOps.contain(img_pil, (img_size,img_size)) # keep the same aspect ratio

    # Keep a copy for visualisations.
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size,img_size), color=(255, 255, 255))
    img_pil = ImageOps.pad(img_pil, size=(img_size,img_size)) # pad with zero on the smallest side

    # Go to numpy 
    resize_img = np.asarray(img_pil)

    # Normalize and go to torch.
    resize_img = normalize_rgb(resize_img)
    x = torch.from_numpy(resize_img).unsqueeze(0).to(device)
    return x, img_pil_full

def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device('cuda')):
    """ Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = torch.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0,0], K[1,1] = focal, focal

    # Set principal point
    if p_x is not None and p_y is not None:
            K[0,-1], K[1,-1] = p_x * img_size, p_y * img_size
    else:
            K[0,-1], K[1,-1] = img_size//2, img_size//2

    # Add batch dimension
    K = K.unsqueeze(0).to(device)
    return K

def load_model(model_name, device=torch.device('cuda')):
    """ Open a checkpoint, build Multi-HMR using saved arguments, load the model weigths. """
    # Model
    ckpt_path = os.path.join(CACHE_DIR_MULTIHMR, model_name+ '.pt')
    if not os.path.isfile(ckpt_path):
        os.makedirs(CACHE_DIR_MULTIHMR, exist_ok=True)
        print(f"{ckpt_path} not found...")
        print("It should be the first time you run the demo code")
        print("Downloading checkpoint from NAVER LABS Europe website...")
        try:
            os.system(f"wget -O {ckpt_path} http://download.europe.naverlabs.com/multihmr/{model_name}.pt")
            print(f"Ckpt downloaded to {ckpt_path}")
        except:
            assert "Please contact fabien.baradel@naverlabs.com or open an issue on the github repo"

    # Load weights
    print("Loading model")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Get arguments saved in the checkpoint to rebuild the model
    kwargs = {}
    for k,v in vars(ckpt['args']).items():
            kwargs[k] = v
    # Inject missing keys with defaults for backward compatibility
    kwargs.setdefault('simple_depth_encoding', 1)
    # --- POSE PARAMETERIZATION EXPERIMENT ---
    # Checkpoints trained with --pose_param carry it in ckpt['args'] and it
    # flows into the model automatically. Old checkpoints (pre-flag) default
    # to 'local-bone', which was the implicit behavior before.
    kwargs.setdefault('pose_param', 'local-bone')
    print(f"pose_param from checkpoint: {kwargs['pose_param']}")
    kwargs['person_center'] = 'head'  # ← override pelvis → head (Anny uses 'head' not 'pelvis')

    # Build the model.
    #
    # CLASS ROUTING - BY CHECKPOINT CONTENT, NOT FILENAME.
    # Two different classes exist in this repo and their weights are NOT
    # interchangeable (different module names, so strict=False loads ~nothing
    # and the head runs with RANDOM weights -> every person renders in the same
    # init pose [root pi/2, all joints identity]):
    #   - Model        (model.py)                  keys: backbone.encoder.*, x_attention_head.*, mlp_classif.*
    #     -> what train.py produces (all anny_s1_*/anny_s2_*/anny_full_run_* checkpoints)
    #   - Multi_HMR    (multi_hmr_anny/multi_hmr.py) keys: encoder.backbone.*, decoder.*, mlp_pose.*
    #     -> the ORIGINAL repo checkpoints (e.g. multiHMR_672_L_anny)
    # The old name-based routing ('anny' in path) sent train.py checkpoints to
    # Multi_HMR, which is why all demo meshes shared one identical pose.
    sd = ckpt['model_state_dict']
    n_trainpy_style = sum(1 for k in sd if k.startswith(('backbone.encoder.', 'x_attention_head.', 'mlp_classif.')))
    n_origrepo_style = sum(1 for k in sd if k.startswith(('encoder.backbone.', 'decoder.', 'dec_to_token.', 'mlp_pose.')))

    if n_trainpy_style > n_origrepo_style:
        print(f"[load_model] checkpoint keys match Model (model.py): "
              f"{n_trainpy_style} vs {n_origrepo_style} -> building Model")
        model = Model(**kwargs).to(device)
    elif n_origrepo_style > 0:
        print(f"[load_model] checkpoint keys match Multi_HMR (original repo): "
              f"{n_origrepo_style} vs {n_trainpy_style} -> building Multi_HMR")
        # Multi_HMR-specific arch params (the original checkpoints were trained
        # with xat_dim=1024; these kwargs are meaningless for Model).
        kwargs['xat_dim'] = 1024
        kwargs['xat_mlp_dim'] = 4 * 1024
        model = ModelAnny(**kwargs).to(device)
    else:
        # Neither anny-style layout: original SMPL-X path.
        kwargs['type'] = getattr(ckpt['args'], 'train_return_type', 'smplx')
        kwargs['img_size'] = ckpt['args'].img_size if isinstance(ckpt['args'].img_size, int) else ckpt['args'].img_size[0]
        model = Model(**kwargs).to(device)

    # Load weights into model - VERIFIED. strict=False silently drops every
    # mismatched key, so count what actually landed and refuse to run a model
    # whose weights did not load (it would silently output the init pose for
    # every person, which looks like a trained-but-broken model).
    msd = model.state_dict()
    matched = [k for k, v in sd.items()
               if k in msd and hasattr(v, 'shape') and msd[k].shape == v.shape]
    model.load_state_dict(sd, strict=False)
    pct = 100.0 * len(matched) / max(len(msd), 1)
    print(f"[load_model] loaded {len(matched)}/{len(msd)} model tensors ({pct:.1f}%)")
    if pct < 80.0:
        unexpected = [k for k in sd if k not in msd][:5]
        raise RuntimeError(
            f"Only {pct:.1f}% of {type(model).__name__} was initialised from "
            f"{ckpt_path}. The checkpoint most likely belongs to the other model "
            f"class (unexpected key examples: {unexpected}). Refusing to run "
            f"inference with mostly-random weights."
        )

    # --- POSE PARAMETERIZATION SAFETY CHECK ---
    # If the checkpoint was trained with a non-default parameterization but
    # the model class silently ignored the kwarg (e.g. swallowed by **kwargs),
    # the predicted rotations would be interpreted in the WRONG space and the
    # meshes would be garbage with no error. Fail loudly instead.
    _ckpt_pp = kwargs.get('pose_param', 'local-bone')
    _model_pp = getattr(model, 'pose_param', None)
    if _ckpt_pp != 'local-bone' and _model_pp != _ckpt_pp:
        raise RuntimeError(
            f"Checkpoint was trained with pose_param='{_ckpt_pp}' but the model "
            f"class {type(model).__name__} does not expose/honor it "
            f"(model.pose_param={_model_pp}). Apply the pose_param edits to "
            f"{type(model).__module__} before running inference with this checkpoint."
        )

    return model

def filter_humans(humans, max_persons=0, iou_thresh=0.45, min_depth_m=0.5, max_depth_m=40.0):
    """Demo-time detection pruning for out-of-domain images.

    1. Rank detections by confidence (requires 'score' from multi_hmr.py).
    2. Greedy NMS on projected-2D bounding-box IoU: overlapping bodies keep
       only the highest-confidence one. This is what collapses the 'wall of
       119 bodies' — the model's grid NMS only suppresses adjacent heatmap
       cells, not bodies whose projections overlap from different cells.
    3. Sanity-band on absolute depth (translation is in cm units).
    4. Hard cap at max_persons (0 = unlimited).
    """
    if len(humans) == 0:
        return humans

    def _score(h):
        # Multi_HMR persons carry 'score'; Model (model.py) persons carry 'scores'.
        s = h.get('score', None)
        if s is None:
            s = h.get('scores', None)
        try:
            return float(s) if s is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _bbox(h):
        j2d = h['j2d'].detach().cpu().numpy()
        return j2d[:, 0].min(), j2d[:, 1].min(), j2d[:, 0].max(), j2d[:, 1].max()

    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    ranked = sorted(humans, key=_score, reverse=True)
    kept, kept_boxes = [], []
    n_drop_depth, n_drop_iou = 0, 0
    z_seen = []
    for h in ranked:
        # Depth sanity band. UNITS ARE AMBIGUOUS: the detector-guided (real
        # photo) path yields translations in centimetre-like units, while the
        # plain in-domain path yields metres. Dividing by 100 unconditionally
        # dropped EVERY in-domain detection (all 5 AnnyOne images -> 0 kept,
        # blank renders). Accept if EITHER reading lands in the band.
        z_raw = float(h['transl'][2])
        z_seen.append(z_raw)
        z_as_m, z_as_cm = abs(z_raw), abs(z_raw) / 100.0
        if not ((min_depth_m <= z_as_m <= max_depth_m) or
                (min_depth_m <= z_as_cm <= max_depth_m)):
            n_drop_depth += 1
            continue
        box = _bbox(h)
        if any(_iou(box, kb) > iou_thresh for kb in kept_boxes):
            n_drop_iou += 1
            continue
        kept.append(h)
        kept_boxes.append(box)
        if max_persons and len(kept) >= max_persons:
            break
    print(f"[filter_humans] {len(humans)} detections -> {len(kept)} kept "
          f"(iou_thresh={iou_thresh}, max_persons={max_persons or 'inf'}; "
          f"dropped: depth={n_drop_depth} iou={n_drop_iou})")
    if len(kept) == 0 and len(humans) > 0:
        print(f"[filter_humans] WARNING: all dropped. raw transl z values: "
              f"{[round(z,2) for z in z_seen[:8]]} "
              f"(band {min_depth_m}-{max_depth_m} m, checked as both m and cm)")
    return kept


_PERSON_DETECTOR = None

def get_person_detector(device):
    """Lazy-load a COCO-pretrained person detector (torchvision Faster R-CNN)."""
    global _PERSON_DETECTOR
    if _PERSON_DETECTOR is None:
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
            w = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            _PERSON_DETECTOR = fasterrcnn_resnet50_fpn(weights=w).to(device).eval()
        except ImportError:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            _PERSON_DETECTOR = fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()
    return _PERSON_DETECTOR


def detector_guided_inference(model, faces, img_path, save_fn, args, device):
    """OOD-photo mode. The Anny model was trained only on synthetic renders, so
    on real photos its own detector fires on texture instead of people. Here a
    COCO-pretrained detector (trained on real photos) proposes person boxes;
    each person is cropped to a square with margin, resized to the network
    input size (big, centered person = close to training distribution), run
    through the Anny model independently, rendered on the crop, and pasted
    back into the full image. Placement + scale are anchored by the 2D
    detector; the Anny model only has to solve pose and shape."""
    import torchvision.transforms.functional as TF

    detector = get_person_detector(device)
    img_full = Image.open(img_path)
    img_full = ImageOps.exif_transpose(img_full).convert('RGB')
    W, H = img_full.size

    with torch.no_grad():
        det = detector([TF.to_tensor(img_full).to(device)])[0]
    keep = (det['labels'] == 1) & (det['scores'] >= args.detector_thresh)
    boxes = det['boxes'][keep].detach().cpu().numpy()
    scores = det['scores'][keep].detach().cpu().numpy()
    order = np.argsort(-scores)
    if args.max_persons:
        order = order[:args.max_persons]
    boxes, scores = boxes[order], scores[order]
    print(f"[detector] {len(scores)} person boxes kept (thresh={args.detector_thresh}): "
          f"scores={np.round(scores, 2).tolist()}")

    canvas = np.asarray(img_full).copy()
    img_size = model.img_size

    for _pi, (b, ds) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = b
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # Crop with CONTEXT so the person appears at training scale.
        # AnnyOne people occupy ~20-30% of the frame; frame-filling crops are
        # so far out of distribution that the root-orientation prediction
        # collapses to the Anny default (bodies render lying sideways).
        S = int(round(max((y2 - y1) / max(args.person_frac, 0.05),
                          max(x2 - x1, y2 - y1) * (1.0 + args.crop_margin))))
        S = max(S, 32)
        # Clamp: when a person fills the frame, person_frac would demand a crop
        # several times the image size. Beyond ~3x it is almost entirely black
        # padding (the person shrinks below training scale anyway) and the
        # allocation becomes huge, so cap it.
        S_max = int(3 * max(W, H))
        if S > S_max:
            print(f"[crop] S={S} clamped to {S_max} (person fills frame; "
                  f"effective person_frac={(y2 - y1) / S_max:.3f})")
            S = S_max
        left, top = int(round(cx - S / 2.0)), int(round(cy - S / 2.0))
        crop = img_full.crop((left, top, left + S, top + S))  # out-of-bounds pads black

        # --- network input: square crop resized to model input size ---
        crop_net = crop.resize((img_size, img_size), Image.BILINEAR)
        x = normalize_rgb(np.asarray(crop_net))
        x = torch.from_numpy(x).unsqueeze(0).to(device)
        K = get_camera_parameters(img_size, fov=args.fov, device=device)

        humans = []
        for _th in [args.det_thresh, 0.05]:
            humans = forward_model(model, x, K,
                                   det_thresh=_th,
                                   nms_kernel_size=args.nms_kernel_size)
            if len(humans) > 0:
                break
        if len(humans) == 0:
            print(f"[detector] box (score {ds:.2f}): Anny found nobody in crop, skipped")
            continue
        # The person is CENTERED in the crop by construction, so pick the
        # detection closest to the crop center (not max score: on OOD crops
        # the score peak can sit on texture like sunglasses or logos).
        _ctr = img_size / 2.0
        best = min(humans, key=lambda h: float(((h['loc'][0] - _ctr) ** 2 +
                                                (h['loc'][1] - _ctr) ** 2)))

        # --- anchor scale + placement to the detector box ---------------
        # The model's depth prior comes from training scenes where people
        # occupy ~20% of the frame, so inside a person-filling crop it
        # renders them tiny. We know the true pixel height (the detector
        # box), so keep the model's POSE+SHAPE but re-place the mesh:
        # centroid on the crop's optical axis at the depth that makes the
        # projected height match the box height.
        if getattr(args, 'anchor_to_box', 1):
            vkey = 'v3d' if 'v3d' in best else ('verts_smplx' if 'verts_smplx' in best else None)
            if vkey is not None:
                v = best[vkey]
                c = v.mean(dim=0)
                h_phys = float(torch.clamp(v[:, 1].max() - v[:, 1].min(), min=1e-3))
                f_net = float(K[0, 0, 0])
                target_h_net = (y2 - y1) / float(S) * img_size   # box height in network px
                z_geom = f_net * h_phys / max(target_h_net, 1e-3)
                # If z_geom is small (big person in frame), the body's own
                # depth (~h_phys) rivals the camera distance and perspective
                # explodes (balloon heads). Clamp the render distance and
                # uniformly scale the mesh instead: projected size is
                # preserved exactly (scale and distance cancel), perspective
                # becomes mild.
                z_render = max(z_geom, 3.0 * h_phys)
                k = z_render / z_geom
                vv = (v - c) * k
                # Lateral centering: align the mesh's BOUNDING-BOX midpoint
                # (not the vertex centroid, which asymmetric poses drag
                # sideways) with the crop center. The overlay's centroid
                # unit-hack divides the mesh mean position by 100, so a
                # lateral correction must be amplified x100 to survive it.
                bbx = (vv[:, 0].max() + vv[:, 0].min()) / 2.0
                bby = (vv[:, 1].max() + vv[:, 1].min()) / 2.0
                new_c = torch.stack([-100.0 * bbx,
                                     -100.0 * bby,
                                     torch.as_tensor(z_render * 100.0,
                                                     device=v.device, dtype=v.dtype)])
                print(f"[anchor v3] z_render={z_render:.2f}m k={k:.2f} "
                      f"bb_mid=({float(bbx):+.3f},{float(bby):+.3f})m")
                best = dict(best)
                best[vkey] = vv + new_c

        # --- render on the native-resolution crop ---
        K_vis = K.clone()
        ratio = S / float(img_size)
        K_vis[0, [0, 1], [0, 1]] = ratio * K_vis[0, [0, 1], [0, 1]]
        K_vis[0, 0, 2] = S / 2.0
        K_vis[0, 1, 2] = S / 2.0
        _person_color = [color[_pi % len(color)]]
        rend, _ = overlay_human_meshes([best], faces, K_vis, model, crop,
                                       alpha=args.alpha, _color=_person_color,
                                       white_bg=False)
        rend = np.asarray(rend).astype(np.uint8)

        # --- paste back ONLY where the mesh changed pixels, so overlapping
        # detector boxes don't erase each other's meshes with background ---
        crop_np = np.asarray(crop).astype(np.uint8)
        mesh_mask = (np.abs(rend.astype(np.int16) - crop_np.astype(np.int16)).sum(-1) > 8)
        sx1, sy1 = max(0, left), max(0, top)
        sx2, sy2 = min(W, left + S), min(H, top + S)
        sub_r = rend[sy1 - top:sy2 - top, sx1 - left:sx2 - left]
        sub_m = mesh_mask[sy1 - top:sy2 - top, sx1 - left:sx2 - left]
        region = canvas[sy1:sy2, sx1:sx2]
        region[sub_m] = sub_r[sub_m]
        canvas[sy1:sy2, sx1:sx2] = region

    _img = np.concatenate([np.asarray(img_full), canvas], 1).astype(np.uint8)
    Image.fromarray(_img).save(save_fn)
    print(f"[detector-guided] saved ---> {save_fn}")


def forward_model(model, input_image, camera_parameters,
                  det_thresh=0.3,
                  nms_kernel_size=1,
                 ):
        
    """ Make a forward pass on an input image and camera parameters. """
    
    # Forward the model.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            # print(model.backbone.encoder.patch_embed.proj.bias.dtype, input_image.dtype)
            humans = model(input_image, 
                           is_training=False, 
                           nms_kernel_size=int(nms_kernel_size),
                           det_thresh=det_thresh,
                           K=camera_parameters)

    # Diagnostic: show what the shape head predicts, so shape-driven artifacts
    # (e.g. chest from cupsize/firmness) can be distinguished from pose-driven
    # ones. Anny phenotypes live in [0,1]; 0.5 is neutral.
    try:
        labels = list(model.body_model.phenotype_labels)
        for hi, h in enumerate(humans):
            vals = h['shape'].detach().float().cpu().reshape(-1).tolist()
            pretty = "  ".join(f"{k}={v:.2f}" for k, v in zip(labels, vals))
            print(f"[shape] person {hi}: {pretty}")
    except Exception as e:
        print(f"[shape] could not print phenotypes: {e}")
    return humans

def overlay_human_meshes(humans, faces, K, model, img_pil, unique_color=False, alpha=0.8, _color=None, white_bg=False):
    if _color is None:
        if unique_color:
            _color = [color[0] for _ in range(len(humans))]
        else:
            _color = [color[j % len(color)] for j in range(len(humans))]

    focal = np.asarray([K[0,0,0].cpu().numpy(), K[0,1,1].cpu().numpy()])
    princpt = np.asarray([K[0,0,-1].cpu().numpy(), K[0,1,-1].cpu().numpy()])

    bg = np.asarray(img_pil).copy()
    if white_bg:
        bg[:] = 255
    pred_rend_array = bg.copy()

    if len(humans) == 0:
        return pred_rend_array, _color

    try:
        verts_list = []
        keep_colors = []

        for j in range(len(humans)):
            if 'verts_smplx' in humans[j]:
                v = humans[j]['verts_smplx']
            elif 'v3d' in humans[j]:
                v = humans[j]['v3d']
            else:
                continue

            v = v.detach().cpu().numpy()

            # if 'transl' in humans[j] and humans[j]['transl'] is not None:
            #     t = humans[j]['transl'].detach().cpu().numpy().reshape(1, 3)
            #     if v.ndim == 2 and v.shape[1] == 3:
            #         v = v + t

            if not np.isfinite(v).all():
                continue
            if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
                continue

            verts_list.append(v.astype(np.float32))
            keep_colors.append(_color[j])

        if len(verts_list) > 0:
            faces_list = [faces for _ in range(len(verts_list))]

            for i, v in enumerate(verts_list):
                print(
                    f"mesh {i}: x[{v[:,0].min():.2f},{v[:,0].max():.2f}] "
                    f"y[{v[:,1].min():.2f},{v[:,1].max():.2f}] "
                    f"z[{v[:,2].min():.2f},{v[:,2].max():.2f}] mean_z={v[:,2].mean():.2f}"
                )

            print("focal:", focal, "princpt:", princpt, "img shape:", bg.shape)

            h, w = bg.shape[:2]

            # Anny meshes come out in a large unit (vertices at z ~ 400-500,
            # x/y in the +-280 range) rather than metres. Convert to metres.
            # Only the VERTICES get scaled — NOT the focal length: in a pinhole
            # camera pixel = focal * X/Z, and scaling X,Y,Z together leaves X/Z
            # unchanged, so focal must stay in pixels. (The old code divided
            # focal by 500 too, which shrank the projection to a sub-pixel dot
            # -> empty render, depth 0.0, nonzero alpha 0.)
            # DIAGNOSED (see mesh stats in log): the mesh GEOMETRY is already
            # metric (extent ~1.5 m per person) but the TRANSLATION is ~100x
            # too large (z ~ 730-820, i.e. centimetres: 7.3-8.2 m -> plausible).
            # So rescale ONLY the position (centroid), never the mesh size:
            #   v_render = (v - c) + c / transl_scale
            # Dividing the whole vertex array (old code) shrank people to
            # 1.7 cm dots; not dividing put 1.7 m people at 780 m -> also dots.
            # UNITS ARE PATH-DEPENDENT (this bit us twice):
            #   real-photo / detector-guided path -> translation in cm (z ~ 400)
            #   in-domain / plain path            -> translation in m  (z ~ 2.8)
            # Unconditionally dividing by 100 put in-domain people 2.8 cm from
            # the lens, so one body filled the whole frame ("giant blob").
            # Detect: a human centroid depth > 30 in *some* unit can only be
            # centimetres (nobody is 30 m away in these images).
            _z_all = [float(v.mean(axis=0)[2]) for v in verts_list]
            _med_z = float(np.median(np.abs(_z_all))) if _z_all else 0.0
            transl_scale = 100.0 if _med_z > 30.0 else 1.0
            print(f"[render] median centroid depth={_med_z:.2f} -> "
                  f"treating translation as {'cm (/100)' if transl_scale == 100.0 else 'metres (no scaling)'}")
            scaled_verts_list = []
            for v in verts_list:
                c = v.mean(axis=0, keepdims=True)          # mesh centroid
                scaled_verts_list.append(v - c + c / transl_scale)

            fx = float(focal[0])          # keep focal in PIXELS
            fy = float(focal[1])
            cx = float(princpt[0])
            cy = float(princpt[1])

            scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])
            _normals_fixed_faces = None
            for v, c in zip(scaled_verts_list, keep_colors):
                mesh_tm = trimesh.Trimesh(vertices=v, faces=faces_list[0], process=False)
                # The quad->triangle split leaves inconsistent winding, so many
                # face normals point inward and shade to near-black even with
                # doubleSided. Repair the winding once (same topology for every
                # person) and reuse the fixed face order for all meshes.
                if _normals_fixed_faces is None:
                    trimesh.repair.fix_normals(mesh_tm)
                    _normals_fixed_faces = mesh_tm.faces.copy()
                else:
                    mesh_tm = trimesh.Trimesh(vertices=v, faces=_normals_fixed_faces, process=False)
                # demo_color from the original Multi-HMR utils is 0-1 floats;
                # dividing those by 255 made every body black. Auto-detect:
                c_arr = np.asarray(c, dtype=np.float64).reshape(-1)[:3]
                if c_arr.max() > 1.0:          # 0-255 int palette
                    c_arr = c_arr / 255.0
                color_rgba = [float(c_arr[0]), float(c_arr[1]), float(c_arr[2]), 1.0]
                mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=color_rgba, metallicFactor=0.0, roughnessFactor=0.8, doubleSided=True)
                mesh_pr = pyrender.Mesh.from_trimesh(mesh_tm, material=mat, smooth=True)
                scene.add(mesh_pr)

            # znear/zfar in the SAME (metre) units as the scaled mesh. With
            # meshes at z ~ 4-5 m after /100, a 0.01..100 range is safe.
            camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
            cam_pose = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float64)
            scene.add(camera, pose=cam_pose)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            scene.add(light, pose=cam_pose)

            r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
            rend_color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            r.delete()
            print(f"depth range: {depth.min():.1f} {depth.max():.1f}, nonzero alpha: {(rend_color[:,:,3]>0).sum()}")

            alpha_mask = (rend_color[:, :, 3:4] / 255.0).astype(np.float32)
            pred_rend_array = (rend_color[:, :, :3] * alpha_mask + bg * (1 - alpha_mask)).astype(np.uint8)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Rendering error:", e)
        if len(humans) > 0:
            print("Human keys:", humans[0].keys())

    return pred_rend_array, _color

def _generate_rotated_frames(humans, faces, K, model, img, center, name, n_frames, angle_range, axis, unique_color, alpha, _color):
    if len(humans) == 0:          # ← ADD THIS
        return []                 # ← ADD THIS    
    frames = []
    for i in range(n_frames):
        angle = angle_range * i / (n_frames - 1)
        theta = np.deg2rad(angle)
        if axis == 'y':
            rotmat = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif axis == 'x':
            rotmat = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        else:
            raise ValueError("Axis must be 'x' or 'y'")
        _humans = []
        for k in range(len(humans)):
            v = get_mesh_verts(humans[k], name)
            if v is None:
                continue

            x = (v - center) @ rotmat.T + center
            _human_k = {name: torch.tensor(x.astype(np.float32))}
            _humans.append(_human_k)
        frame, _ = overlay_human_meshes(_humans, faces, K, model, img, unique_color=unique_color, alpha=alpha, _color=_color)
        frames.append(frame.astype(np.uint8))
    return frames

def create_rotating_video(humans, faces, K, model, img_pil_visu, unique_color=False, alpha=0.8, fn='rotating.mp4', n_frames=20, angle_range=60):
    if len(humans) == 0:
        return None

    central, _color = overlay_human_meshes(
    humans, faces, K, model, img_pil_visu,
    unique_color=unique_color, alpha=alpha, _color=None, white_bg=True)
    white_img = Image.new(img_pil_visu.mode, img_pil_visu.size, (255, 255, 255))

    # Pick the right key — prefer surface mesh over joints
    if 'verts_smplx' in humans[0]:
        name = 'verts_smplx'
    elif 'verts' in humans[0]:
        name = 'verts'
    elif 'vertices' in humans[0]:
        name = 'vertices'
    else:
        name = 'v3d'  # fallback: joints only, will look wrong

    valid_idx = []
    for j, h in enumerate(humans):
        v = get_mesh_verts(h, 'verts_smplx' if 'verts_smplx' in h else 'v3d')
        if v is None:
            continue
        z = float(v[:, 2].mean())
        if z > 0:
            valid_idx.append((z, j))

    if not valid_idx:
        return None

    closest_idx = min(valid_idx)[1]
    center = get_mesh_verts(humans[closest_idx], name).mean(0)

    closest_idx = min(valid_idx)[1]
    center = humans[closest_idx][name].detach().cpu().numpy().mean(0)

    frames_central_to_right = _generate_rotated_frames(humans, faces, K, model, white_img, center, name, n_frames, angle_range, 'y', unique_color, alpha, _color)
    frames_right_to_central = frames_central_to_right[::-1][1:-1]
    frames_central_to_left = _generate_rotated_frames(humans, faces, K, model, white_img, center, name, n_frames, -angle_range, 'y', unique_color, alpha, _color)
    frames_left_to_central = frames_central_to_left[::-1][1:-1]
    frames_central_to_top = _generate_rotated_frames(humans, faces, K, model, white_img, center, name, n_frames, angle_range, 'x', unique_color, alpha, _color)
    frames_top_to_central = frames_central_to_top[::-1][1:-1]

    print(len(frames_central_to_right))
    print(len(frames_right_to_central))
    print(len(frames_central_to_left))
    print(len(frames_left_to_central))
    print(len(frames_central_to_top))
    print(len(frames_top_to_central))

    frames = [central.astype(np.uint8) for _ in range(n_frames//4)] + \
            frames_central_to_right + \
            frames_right_to_central + \
            [central.astype(np.uint8) for _ in range(n_frames//4)] + \
            frames_central_to_left + \
            frames_left_to_central + \
            [central.astype(np.uint8) for _ in range(n_frames//4)] + \
            frames_central_to_top + \
            frames_top_to_central + \
            [central.astype(np.uint8) for _ in range(n_frames//4)]

    # Create a video from frames
    if fn is not None:
        import cv2
        height, width, layers = frames[0].shape
        fps = 10
        video_path = fn
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for frame in tqdm(frames, desc=f"Writing video"):
            img = frame[:, :, ::-1]  # Convert RGB to BGR
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            video.write(img)

        video.release()
        print(f"Saved video to {video_path}")

    return frames

def get_mesh_verts(h, key='v3d'):
    if key not in h:
        return None

    v = h[key]
    if v is None or not torch.isfinite(v).all():
        return None

    v = v.detach().cpu().numpy()
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] == 0:
        return None

    if 'transl' in h and h['transl'] is not None:
        t = h['transl'].detach().cpu().numpy().reshape(1, 3)
        v = v + t

    if not np.isfinite(v).all():
        return None

    return v.astype(np.float32)

if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--model_name", type=str, default='multiHMR_896_L_synth')
        parser.add_argument("--img_folder", type=str, default='example_data')
        parser.add_argument("--out_folder", type=str, default='demo_out')
        parser.add_argument("--save_mesh", type=int, default=0, choices=[0,1])
        parser.add_argument("--use_person_detector", type=int, default=0, choices=[0,1], help="OOD mode: COCO-pretrained detector proposes person boxes; Anny model runs per crop and results are pasted back")
        parser.add_argument("--detector_thresh", type=float, default=0.8, help="min confidence for detector person boxes")
        parser.add_argument("--crop_margin", type=float, default=0.35, help="margin added around each person box before cropping")
        parser.add_argument("--anchor_to_box", type=int, default=1, choices=[0,1], help="rescale/re-place each crop mesh so it fills the detector box (pose+shape from model, placement from detector)")
        parser.add_argument("--person_frac", type=float, default=0.30, help="fraction of the crop the person should occupy (~training scale). Lower = more context")
        parser.add_argument("--extra_views", type=int, default=0, choices=[0,1])
        parser.add_argument("--save_rotating_video", type=int, default=0, choices=[0,1])        
        parser.add_argument("--det_thresh", type=float, default=0.3)
        parser.add_argument("--nms_kernel_size", type=float, default=3)
        parser.add_argument("--fov", type=float, default=60)
        parser.add_argument("--max_persons", type=int, default=0, help="keep only the K most confident detections (0 = unlimited)")
        parser.add_argument("--iou_thresh", type=float, default=0.45, help="2D bbox IoU above which overlapping detections are suppressed (keep highest confidence)")
        parser.add_argument("--distance", type=int, default=0, choices=[0,1], help='add distance on the reprojected mesh')
        parser.add_argument("--unique_color", type=int, default=0, choices=[0,1], help='only one color for all humans')
        parser.add_argument("--alpha", type=float, default=1.0, help='alpha blending value for rendering')
        parser.add_argument("--mask_helper_joints", type=int, default=1, choices=[0, 1],
                            help="replace Anny helper/deform bone rotations (breast, spine/neck "
                                 "subdivisions, face micro-bones) with identity, as the original "
                                 "repo does via useful_rotmat. Fixes chest crease / pinched neck "
                                 "caused by noisy predictions on bones the network cannot learn. "
                                 "Set 0 to see the raw unmasked prediction.")
        parser.add_argument("--neutral_shape", type=int, default=0, choices=[0, 1],
                            help="force all 11 phenotypes to 0.5 at render time. Diagnostic: if a "
                                 "deformation persists with this on, it comes from pose, not shape.")
        parser.add_argument("--relax_hands", type=int, default=0, choices=[0, 1],
                            help="force finger/thumb bones (found by name) to rest pose at "
                                 "inference. Fingers are undertrained at 672px and often render "
                                 "claw-like; this trades predicted fingers for clean neutral hands.")
        parser.add_argument("--neutral_phenotypes", type=str, default="",
                            help="comma-separated phenotype names to force to 0.5 at render time, "
                                 "keeping all others predicted. E.g. 'proportions,age' to remove "
                                 "the systematic out-of-domain bias observed on real photos.")
        parser.add_argument("--rest_pose", type=int, default=0, choices=[0, 1],
                            help="DIAGNOSTIC: render every person in Anny's neutral standing pose "
                                 "(all joints identity except root) with the predicted shape and "
                                 "placement. If a deformation (e.g. long neck) persists here, it is "
                                 "shape/body-model driven, NOT a pose prediction error.")
        parser.add_argument("--fov_json", type=str, default="",
                            help="path to a JSON file mapping image filename -> true FOV in "
                                 "degrees (written by the AnnyOne export script). Overrides --fov "
                                 "per image. Essential for AnnyOne, where every image has a "
                                 "different camera.")
        parser.add_argument("--no_filter", type=int, default=0, choices=[0, 1],
                            help="bypass filter_humans entirely. Its depth band and IoU NMS are "
                                 "tuned for the detector-guided real-photo path; on the in-domain "
                                 "path they can drop every detection. Use 1 for diagnostic runs.")

        args = parser.parse_args()

        # Load the optional per-image FOV map (see --fov_json).
        _FOV_MAP = {}
        if getattr(args, 'fov_json', '') and os.path.isfile(args.fov_json):
            import json as _json
            with open(args.fov_json) as _f:
                _FOV_MAP = _json.load(_f)
            print(f"[fov] loaded per-image FOV map with {len(_FOV_MAP)} entries "
                  f"from {args.fov_json}")
        elif getattr(args, 'fov_json', ''):
            print(f"[fov] WARNING: --fov_json {args.fov_json} not found; using global --fov {args.fov}")

        dict_args = vars(args)

        assert torch.cuda.is_available()

        is_anny = 'anny' in args.model_name.lower()

        if not is_anny:
            # SMPL-X models
            smplx_fn = os.path.join(SMPLX_DIR, 'smplx', 'SMPLX_NEUTRAL.npz')
            if not os.path.isfile(smplx_fn):
                print(f"{smplx_fn} not found, please download SMPLX_NEUTRAL.npz file")
                print("To do so you need to create an account in https://smpl-x.is.tue.mpg.de")
                print("Then download 'SMPL-X-v1.1 (NPZ+PKL, 830MB) - Use thsi for SMPL-X Python codebase'")
                print(f"Extract the zip file and move SMPLX_NEUTRAL.npz to {smplx_fn}")
                print("Sorry for this incovenience but we do not have license for redustributing SMPLX model")
                assert NotImplementedError
            else:
                print('SMPLX found')
                
            # SMPL mean params download
            if not os.path.isfile(MEAN_PARAMS):
                print('Start to download the SMPL mean params')
                os.system(f"wget -O {MEAN_PARAMS}  https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4")
                print('SMPL mean params have been succesfully downloaded')
            else:
                print('SMPL mean params is already here')

        # Input images
        suffixes = ('.jpg', '.jpeg', '.png', '.webp')
        if os.path.isfile(args.img_folder) and args.img_folder.lower().endswith(suffixes):
            l_img_path = [os.path.basename(args.img_folder)]
            args.img_folder = os.path.dirname(args.img_folder)
        else:
            l_img_path = []
            img_root = os.path.abspath(args.img_folder)
            for root, _, files in os.walk(img_root):
                for fname in files:
                    if fname.lower().endswith(suffixes) and not fname.startswith('.'):
                        abs_path = os.path.join(root, fname)
                        rel_path = os.path.relpath(abs_path, img_root)
                        l_img_path.append(rel_path)
            l_img_path.sort()

        # Loading
        model = load_model(args.model_name)
        # Inference-time toggles (no effect on training; see Model.__init__).
        if hasattr(model, 'mask_helper_joints'):
            model.mask_helper_joints = bool(args.mask_helper_joints)
            model.neutral_shape = bool(args.neutral_shape)
            if hasattr(model, 'relax_hands'):
                model.relax_hands = bool(args.relax_hands)
                model.neutral_phenotype_names = [
                    s.strip() for s in args.neutral_phenotypes.split(',') if s.strip()
                ]
            if hasattr(model, 'rest_pose'):
                model.rest_pose = bool(args.rest_pose)
            print(f"[demo] mask_helper_joints={model.mask_helper_joints} "
                  f"neutral_shape={model.neutral_shape} "
                  f"relax_hands={getattr(model, 'relax_hands', 'n/a')} "
                  f"neutral_phenotypes={getattr(model, 'neutral_phenotype_names', [])} "
                  f"rest_pose={getattr(model, 'rest_pose', 'n/a')}")
        elif args.mask_helper_joints == 0:
            # Multi_HMR always masks internally; there is no raw-rotation mode there.
            print("[demo] note: this model class always masks helper joints internally")
        # Both anny classes (Model from model.py and Multi_HMR from
        # multi_hmr_anny) carry the anny body model as .body_model; the class
        # itself no longer tells us which body model is in play.
        is_anny = hasattr(model, 'body_model') and getattr(model.body_model, 'name', '') == 'anny'

        if is_anny:
            faces = model.body_model.faces.cpu().numpy().astype(np.int32)
            print(f"faces shape: {faces.shape}, max face index: {faces.max()}, num verts: 19158")
            # Convert quads to triangles if needed
            if faces.shape[1] == 4:
                tri_faces = np.concatenate([
                    faces[:, [0, 1, 2]],
                    faces[:, [0, 2, 3]]
                ], axis=0)
                faces = tri_faces
                print(f"Converted quads to triangles: {faces.shape}")
        
        else:
            faces = model.smpl_layer['neutral_10'].bm_x.faces

        # Model name for saving results.
        model_name = os.path.basename(args.model_name)

        # All images
        os.makedirs(args.out_folder, exist_ok=True)
        l_duration = []
        for i, img_path in enumerate(tqdm(l_img_path)):
            # Compose save_fn: out_folder/rel_path_no_ext_modelname.png
            save_fn = os.path.join(args.out_folder, f"{img_path}_{model_name}.png")
            # Ensure output subdirectories exist
            os.makedirs(os.path.dirname(save_fn), exist_ok=True)

            # OOD-photo mode: detector proposes people, Anny solves each crop
            if args.use_person_detector:
                detector_guided_inference(model, faces,
                                          os.path.join(args.img_folder, img_path),
                                          save_fn, args, torch.device('cuda'))
                continue

            # Get input in the right format for the model
            img_size = model.img_size
            x, img_pil_visu = open_image(os.path.join(args.img_folder, img_path), img_size)

            # Get camera parameters
            p_x, p_y = None, None
            # Per-image field of view. AnnyOne images each have their own
            # camera (measured FOVs on the holdout set ranged 51-110 deg), so a
            # single global --fov mis-places meshes on most of them. When
            # --fov_json points at a {filename: fov_degrees} map written by the
            # export script, use the true value for THIS image.
            _fov = args.fov
            if _FOV_MAP:
                _key = os.path.basename(img_path)
                if _key in _FOV_MAP:
                    _fov = float(_FOV_MAP[_key])
                    print(f"[fov] {_key}: using true fov={_fov:.1f} deg")
                else:
                    print(f"[fov] {_key}: not in fov map, falling back to --fov {_fov}")
            K = get_camera_parameters(model.img_size, fov=_fov, p_x=p_x, p_y=p_y)

            # Make model predictions
            start = time.time()
            humans = forward_model(model, x, K,
                                             det_thresh=args.det_thresh,
                                             nms_kernel_size=args.nms_kernel_size)
            # confidence-ranked pruning for out-of-domain photos.
            # --no_filter 1 bypasses it: the filter is tuned for the
            # detector-guided real-photo path and can wrongly drop everything
            # on the in-domain path.
            if getattr(args, 'no_filter', 0):
                print(f"[filter_humans] BYPASSED (--no_filter 1): keeping all {len(humans)} detections")
            else:
                humans = filter_humans(humans,
                                       max_persons=args.max_persons,
                                       iou_thresh=args.iou_thresh)
            if is_anny and len(humans) > 0:
                print("ALL Anny keys:")
                for k, val in humans[0].items():
                    shape = val.shape if hasattr(val, 'shape') else type(val)
                    print(f"  {k}: {shape}")
                    
            # DEBUG: print what keys Anny returns + vertex/joint sizes
            for j, h in enumerate(humans[:3]):
                dbg_key = 'v3d' if 'v3d' in h else 'verts_smplx' if 'verts_smplx' in h else None
                if dbg_key is None:
                    continue
                v = h[dbg_key]
                print(f"person {j}: nan={torch.isnan(v).any()}, min={v.min():.2f}, max={v.max():.2f}")
                # ---- vertices ----
                _v = v.detach().cpu() if hasattr(v, 'detach') else torch.as_tensor(v)
                _v = _v.reshape(-1, 3)
                v_ext = (_v.max(0).values - _v.min(0).values)
                print(f"  vertices ({dbg_key}): shape={tuple(v.shape)}, "
                      f"extent x={v_ext[0]:.3f} y={v_ext[1]:.3f} z={v_ext[2]:.3f} "
                      f"(height~y-extent)")
                # ---- joints ----
                jk = 'j3d' if 'j3d' in h and h['j3d'] is not None else None
                if jk is not None:
                    jt = h[jk]
                    _j = jt.detach().cpu() if hasattr(jt, 'detach') else torch.as_tensor(jt)
                    _j = _j.reshape(-1, 3)
                    j_ext = (_j.max(0).values - _j.min(0).values)
                    print(f"  joints (j3d): shape={tuple(jt.shape)}, n_joints={_j.shape[0]}, "
                          f"extent x={j_ext[0]:.3f} y={j_ext[1]:.3f} z={j_ext[2]:.3f}")
                else:
                    print("  joints: no 'j3d' key present")
                # ---- 2d joints, if any ----
                if 'j2d' in h and h['j2d'] is not None:
                    print(f"  joints 2d (j2d): shape={tuple(h['j2d'].shape)}")
            
            # print the model's own camera prediction (Multi_HMR persons carry
            # 'fov'/'K_regressed'; Model persons do not - skip in that case)
            if len(humans) > 0 and 'fov' in humans[0] and 'K_regressed' in humans[0]:
                fov_rad = float(humans[0]['fov'].reshape(-1)[0])
                Kr = humans[0]['K_regressed'].reshape(3, 3)
                f_px = float(Kr[0, 0])
                fov_from_K = 2 * math.degrees(math.atan((672 / 2) / f_px))
                print(f"model camera: fov={math.degrees(fov_rad):.1f} deg (raw {fov_rad:.3f} rad), "
                    f"K_regressed focal={f_px:.1f}px -> fov={fov_from_K:.1f} deg")

            if is_anny and len(humans) > 0:
                print("Anny output keys:", list(humans[0].keys()))
                
            duration = time.time() - start
            l_duration.append(duration)

            # Update K for rendering at full resolution
            ratio = max(img_pil_visu.size) / x.shape[-1]
            K[0, 0, 2] = img_pil_visu.size[0] / 2.0
            K[0, 1, 2] = img_pil_visu.size[1] / 2.0
            K[0, [0, 1], [0, 1]] = ratio * K[0, [0, 1], [0, 1]]
             
            v3d_key = 'verts_smplx' if humans and 'verts_smplx' in humans[0] else 'v3d' if humans and 'v3d' in humans[0] else None
            if v3d_key is not None:
                v_dbg = get_mesh_verts(humans[0], v3d_key)
                print("first human keys:", humans[0].keys())
                if v_dbg is not None:
                    print("vertex shape:", v_dbg.shape)
                    print("z range:", float(v_dbg[:, 2].min()), float(v_dbg[:, 2].max()))
                else:
                    print("Mesh key exists but vertices are invalid after filtering.")
            else:
                print("No mesh key found in humans[0]")
            
            if humans:
                print("keys:", humans[0].keys())
                for k in ("v3d", "verts_smplx", "transl"):
                    if k in humans[0]:
                        v = humans[0][k]
                        print(k, v.shape, torch.isfinite(v).all().item(), v.min().item(), v.max().item())
                        
            pred_rend_array, _color = overlay_human_meshes(humans, faces, K, model, img_pil_visu,unique_color=args.unique_color,alpha=args.alpha,white_bg=False)

            # Optionally add distance as an annotation to each mesh
            if args.distance and humans and humans[0].get('j3d') is not None:
                pred_rend_array = print_distance_on_image(pred_rend_array, humans, _color)

            # List of images too view side by side.
            l_img = [np.asarray(img_pil_visu), pred_rend_array]

            # sideview - 45 degress
            if args.extra_views:
                frames = create_rotating_video(humans, faces, K, model, img_pil_visu, unique_color=args.unique_color, alpha=args.alpha, fn=None, n_frames=2, angle_range=30)
                if frames is not None and len(frames) > 1:
                    l_img.append(frames[1])

            # Save to path.
            _img = np.concatenate(l_img, 1).astype(np.uint8)
            Image.fromarray(_img).save(save_fn)
            print(f"Avg Multi-HMR inference time={int(1000*np.median(np.asarray(l_duration[-1:])))}ms on a {torch.cuda.get_device_name()} ---> {save_fn}")
            sys.stdout.flush()

            # video
            if args.save_rotating_video:
                
                if humans and any(h.get(v3d_key) is not None for h in humans):
                    create_rotating_video(humans, faces, K, model, img_pil_visu,
                        unique_color=args.unique_color, alpha=args.alpha,
                        fn=save_fn.replace('.png','_rotating.mp4'), n_frames=20, angle_range=60)
                else:
                    print(f"Skipping rotating video: no humans detected or no mesh in output (det_thresh={args.det_thresh})")

            # Saving mesh
            if args.save_mesh:
                _key = 'verts_smplx' if humans and isinstance(humans[0], dict) and 'verts_smplx' in humans[0] else 'v3d'

                valid_humans = [h for h in humans if isinstance(h, dict) and h.get(_key) is not None]

                if valid_humans:
                    l_mesh = []
                    for hum in valid_humans:
                        v = hum[_key].detach().cpu().numpy()
                        if 'transl' in hum and hum['transl'] is not None:
                            v = v + hum['transl'].detach().cpu().numpy().reshape(1, 3)
                        l_mesh.append(v)

                    mesh_fn = save_fn + '.npy'
                    np.save(mesh_fn, np.asarray(l_mesh, dtype=object), allow_pickle=True)

                    l_face = [faces for _ in range(len(l_mesh))]
                    scene = create_scene(img_pil_visu, l_mesh, l_face, color=None, metallicFactor=0., roughnessFactor=0.5)
                    scene_fn = save_fn + '.glb'
                    scene.export(scene_fn)
                else:
                    print("Skipping mesh saving: no valid mesh dictionaries found.")

        print('end')