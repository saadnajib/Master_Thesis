"""
SMPL -> Anny parameter conversion for 3DPW.

WHAT THIS DOES
--------------
3DPW ships ground truth as SMPL parameters (pose + betas). Our model predicts
Anny parameters (163 bone rotations + 11 phenotypes). To train or evaluate on
3DPW we must express its ground truth in Anny's parameter space.

METHOD (following the references)
---------------------------------
  * NVlabs smpl2soma.py converts SMPL -> another body model by fitting to the
    POSED MESH, using inverse-LBS plus an autograd FK refinement stage, and
    reports per-vertex error.
  * wangsen1312/joints2smpl fits body-model parameters to 3D JOINTS by gradient
    descent with pose/shape priors.

We have no analytical inverse-LBS for Anny, so we use the autograd route:
    1. Run SMPL forward with 3DPW's GT params -> SMPL joints (and vertices)
    2. Optimise Anny's pose + shape so Anny's joints land on SMPL's joints
    3. Report the residual error so bad fits can be discarded

Optimisation is STAGED, which matters a lot for convergence:
    Stage 1  global orientation + translation only  (align the body as a rigid
             object first - optimising limbs while the root is wrong wastes
             effort and lands in bad local minima)
    Stage 2  + body joint rotations
    Stage 3  + shape phenotypes  (shape last: it is the weakest signal and will
             otherwise absorb pose error by deforming the body)

THE JOINT CORRESPONDENCE PROBLEM
--------------------------------
SMPL has 24 joints; Anny has 163 bones with different names and a different
rest pose. The fit is only as good as the mapping between them.

VERIFIED (job 3409000/3409022) against Anny's REAL bone names: all 24 SMPL
joints resolve, hip mapping uses 'pelvis.L'/'pelvis.R' (the true hip-joint
bones under 'root'), not 'upperleg01.L/R' one level further down the chain.

THE 11 PHENOTYPES: NOT MAPPED - OPTIMISED
------------------------------------------
There is no SMPL equivalent of Anny's phenotypes to map from. SMPL's shape
space is 10 abstract PCA "betas" with no semantic meaning; Anny's 11
phenotypes (gender, age, muscle, weight, height, proportions, cupsize,
firmness, african, asian, caucasian) are semantic but have no correspondence
to SMPL's betas. So shape starts at 0.5 (neutral) for all 11 and is optimised
by gradient descent in Stage 3, purely to make Anny's joints match SMPL's -
the one thing directly comparable between the two body models. The fitted
values are printed per frame below so you can sanity-check them (e.g. does
'height' come out sensible for this specific person).

THE NaN BUG (job 3409031) AND ITS FIX
--------------------------------------
Every frame came back NaN from iteration 1 of stage 1 - not degrading into NaN
over iterations, ALREADY NaN at the start. That pattern means the INPUT SMPL
pose for that frame was already NaN, not that the optimiser diverged.

3DPW ships a per-frame validity flag, 'campose_valid': some frames (occlusion,
failed tracking) have garbage or literally NaN pose/translation values. The
previous version ignored this field and sampled frames by a blind fixed
stride (0, 50, 100, 150), which apparently landed on invalid frames.

FIX: filter to frames where campose_valid is true AND the SMPL forward output
contains no NaN, THEN apply the stride within that valid set. If a picked
frame still comes back NaN despite the filter (rare, but bugs happen), it is
skipped and reported rather than silently corrupting the batch.

USAGE
    python smpl_to_anny.py --list_bones               # inspect names
    python smpl_to_anny.py --check_mapping             # verify mapping
    python smpl_to_anny.py --pkl <3dpw.pkl> --smpl_model_path <SMPL_NEUTRAL.pkl> \
                           --out fit.npz --max_frames 4
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import torch

# ---------------------------------------------------------------------------
# SMPL joint order (standard 24-joint skeleton)
# ---------------------------------------------------------------------------
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand',
]

# ---------------------------------------------------------------------------
# SMPL joint -> candidate Anny bone names, in priority order.
# VERIFIED against the real 163-bone list from job 3409000/3409022.
# ---------------------------------------------------------------------------
SMPL_TO_ANNY = {
    'pelvis':         ['root', 'pelvis', 'hips'],
    'left_hip':       ['pelvis.L', 'upperleg01.L', 'thigh.L', 'hip.L'],
    'right_hip':      ['pelvis.R', 'upperleg01.R', 'thigh.R', 'hip.R'],
    'spine1':         ['spine01', 'spine1', 'spine_01'],
    'left_knee':      ['lowerleg01.L', 'shin.L', 'lowerleg.L', 'knee.L'],
    'right_knee':     ['lowerleg01.R', 'shin.R', 'lowerleg.R', 'knee.R'],
    'spine2':         ['spine02', 'spine2', 'spine_02'],
    'left_ankle':     ['foot.L', 'ankle.L'],
    'right_ankle':    ['foot.R', 'ankle.R'],
    'spine3':         ['spine03', 'spine3', 'spine_03', 'chest'],
    'left_foot':      ['toe1-1.L', 'toe.L', 'ball.L'],
    'right_foot':     ['toe1-1.R', 'toe.R', 'ball.R'],
    'neck':           ['neck01', 'neck', 'neck_01'],
    'left_collar':    ['clavicle.L', 'collar.L', 'shoulder01.L'],
    'right_collar':   ['clavicle.R', 'collar.R', 'shoulder01.R'],
    'head':           ['head'],
    'left_shoulder':  ['upperarm01.L', 'upperarm.L', 'shoulder.L'],
    'right_shoulder': ['upperarm01.R', 'upperarm.R', 'shoulder.R'],
    'left_elbow':     ['lowerarm01.L', 'lowerarm.L', 'forearm.L', 'elbow.L'],
    'right_elbow':    ['lowerarm01.R', 'lowerarm.R', 'forearm.R', 'elbow.R'],
    'left_wrist':     ['wrist.L', 'hand.L'],
    'right_wrist':    ['wrist.R', 'hand.R'],
    'left_hand':      ['finger2-1.L', 'metacarpal1.L', 'hand.L'],
    'right_hand':     ['finger2-1.R', 'metacarpal1.R', 'hand.R'],
}

# ---------------------------------------------------------------------------
# PER-JOINT LOSS WEIGHTS  (added after check job 3413964)
# ---------------------------------------------------------------------------
# The 4-frame check showed limbs fitting at 3-13 mm while spine1/spine3 sat at
# ~135 mm and the hips at ~90 mm. Those torso joints are DEFINED in different
# anatomical places in the two rigs (SMPL's spine joints lie deep inside the
# torso and its hip joints are low & lateral; Anny's spine0x/pelvis.L/R are
# elsewhere). No pose can make them coincide, so with weight 1.0 the optimiser
# bent spine01/spine02 by 60-80 deg trying to reach them - a distorted torso for
# a target that is not reachable. Down-weight them so they only give a coarse
# hint and let the well-defined joints (limbs, head, pelvis) drive the fit.
JOINT_WEIGHTS = {
    'spine1': 0.1, 'spine3': 0.1, 'left_hip': 0.2, 'right_hip': 0.2,
    'left_collar': 0.1, 'right_collar': 0.1, 'spine2': 0.5,
}
# (collars 0.3 -> 0.1 in v3: job 3413987 showed the 67 mm collar residual was
#  pulling the clavicle away from where it puts the SHOULDER, which is a
#  reliable joint. The clavicle should be driven by the shoulder target.)
# Joints whose definitions DO agree between the rigs; the error we report as
# "limb_err_mm" is computed over these only and is the honest fit quality.
RELIABLE_JOINTS = [
    'pelvis', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'left_foot', 'right_foot', 'neck', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand',
]

# Phenotypes that actually move joints (bone lengths). muscle/weight/cupsize/
# firmness change the SKIN, not the skeleton, so joints cannot constrain them:
# in job 3413964 muscle/weight swung 0.17<->0.78 between frames of the SAME
# person purely absorbing noise. Everything not listed here stays at 0.5.
SKELETAL_PHENOTYPES = ['gender', 'age', 'height', 'proportions']

# Pose prior: penalises rotation angle^2 (rad^2) per optimised bone.
# v2 (job 3413987) used a single weight 1e-3 on EVERY bone. That fixed the
# folded spine (spine01 66deg -> 8deg) but a 90deg elbow bend then cost the
# same as a 5 cm joint error, so elbows/wrists/hands went from 3-13 mm (v1)
# to 21-26 mm. The prior is only needed where the targets are unreliable
# (torso), so v3 makes it PER BONE, keyed by the SMPL joint the bone carries:
#   torso chain   1e-3   spine cannot fold to chase spine1/spine3
#   clavicles     3e-4   collar target is unreliable too
#   everything    1e-5   limbs are fully constrained by good targets;
#   else                 90deg bend ~= 5 mm, just enough to break ties
POSE_PRIOR_W_DEFAULT = 1e-5
POSE_PRIOR_W = {
    'spine1': 1e-3, 'spine2': 1e-3, 'spine3': 1e-3, 'neck': 1e-3,
    'left_collar': 3e-4, 'right_collar': 3e-4,
}


def build_anny():
    import anny
    m = anny.create_fullbody_model(remove_unattached_vertices=False,
                                   all_phenotypes=True).to(dtype=torch.float32)
    # Mirror the setup that BOTH model.py and multi_hmr_anny/multi_hmr.py run
    # immediately after create_fullbody_model(), before ever calling the model.
    # Skipping it risks the phenotype kwargs not lining up with what the model
    # expects.
    m.shape_keys = [k for k in m.phenotype_labels if k != 'race']
    m.shape_keys.extend(m.phenotype_labels[-3:])
    m.set_skinning_method('lbs')
    m.name = 'anny'
    return m


def resolve_mapping(bone_labels, verbose=True):
    """Map SMPL joint index -> Anny bone index, by name, reporting the result."""
    lower = {n.lower(): i for i, n in enumerate(bone_labels)}
    pairs, unmapped = [], []
    for smpl_idx, smpl_name in enumerate(SMPL_JOINT_NAMES):
        hit = None
        for cand in SMPL_TO_ANNY.get(smpl_name, []):
            if cand.lower() in lower:
                hit = (cand, lower[cand.lower()])
                break
        if hit:
            pairs.append((smpl_idx, hit[1], smpl_name, hit[0]))
        else:
            unmapped.append(smpl_name)

    if verbose:
        print(f"\n{'SMPL joint':<16} -> {'Anny bone':<20} (anny idx)")
        print("-" * 52)
        for si, ai, sn, an in pairs:
            print(f"{sn:<16} -> {an:<20} ({ai})")
        if unmapped:
            print(f"\nUNMAPPED ({len(unmapped)}): {unmapped}")
            print("These are excluded from the fit. If important joints are")
            print("listed, add the correct Anny name to SMPL_TO_ANNY.")
        print(f"\nmapped {len(pairs)}/{len(SMPL_JOINT_NAMES)} SMPL joints")
        if len(pairs) < 12:
            print("WARNING: fewer than 12 joints mapped - the fit will be")
            print("         badly under-constrained. Fix the mapping first.")
    return pairs


def fit_frame(anny_model, target_joints, anny_idx, smpl_idx, device,
              iters=(150, 400, 250), lr=0.05, verbose=False,
              fixed_shape_logit=None, init=None):
    """Optimise Anny pose+shape so its joints match `target_joints`.

    target_joints: [24, 3] SMPL joint positions in metres
    fixed_shape_logit: [11] tensor -> shape is frozen to it and stage 3 is
                       skipped (used for the per-PERSON shape pass).
    init: dict with 'rot6d' [163,6] and 'transl' [3] to warm-start from
          (previous pass or previous frame) instead of the rest pose.
    Returns dict with pose (163,3,3 rotmats), shape (11,), transl (3,), error,
    and the fitted phenotype values by name (see module docstring: shape is
    OPTIMISED, not mapped from anything in SMPL).
    """
    from utils import rotation_to_homogeneous
    import roma

    n_bones = len(anny_model.bone_labels)
    tgt = torch.as_tensor(target_joints, dtype=torch.float32, device=device)
    tgt_sel = tgt[smpl_idx]                       # [M, 3]

    # per-joint weights, in the order of smpl_idx
    w = torch.tensor([JOINT_WEIGHTS.get(SMPL_JOINT_NAMES[i], 1.0) for i in smpl_idx],
                     dtype=torch.float32, device=device)                 # [M]
    reliable = torch.tensor([SMPL_JOINT_NAMES[i] in RELIABLE_JOINTS for i in smpl_idx],
                            dtype=torch.bool, device=device)             # [M]

    # only bones that carry a mapped joint (plus root) may rotate; the other
    # 130+ bones (twist bones, fingers, toes, spine04, neck02/03 ...) are not
    # constrained by any target and in job 3413964 drifted to 20-100 deg.
    bone_mask = torch.zeros(n_bones, 1, device=device)
    bone_mask[0] = 1.0
    bone_mask[list(anny_idx)] = 1.0
    # per-bone prior weight (0 on root and on all non-optimised bones)
    pose_prior_mask = torch.zeros(n_bones, 1, device=device)
    for si, ai in zip(smpl_idx, anny_idx):
        pose_prior_mask[ai, 0] = POSE_PRIOR_W.get(SMPL_JOINT_NAMES[si], POSE_PRIOR_W_DEFAULT)
    pose_prior_mask[0, 0] = 0.0                                          # no prior on root
    shape_mask = torch.tensor([1.0 if k in SKELETAL_PHENOTYPES else 0.0
                               for k in anny_model.phenotype_labels], device=device)

    # 6D IDENTITY ROTATION - indices 0 and 3, NOT 0 and 4.
    # The 6 numbers reshape to a 3x2 matrix taken COLUMN-wise; setting 0 and 4
    # yields columns (1,0,0) and (0,0,1)... actually a DEGENERATE matrix whose
    # Gram-Schmidt divides by zero -> NaN for all 163 bones before a single
    # optimiser step. That was the cause of the all-NaN fits in jobs 3409031
    # and 3413616 (NOT invalid 3DPW frames, and NOT the SMPL forward - both
    # were checked and were clean). Indices 0 and 3 give columns (1,0,0) and
    # (0,1,0) -> a true identity rotation. Verified numerically.
    rot6d = torch.zeros(n_bones, 6, device=device)
    rot6d[:, 0] = 1.0
    rot6d[:, 3] = 1.0
    transl = torch.zeros(3, device=device)
    if init is not None:
        rot6d = torch.as_tensor(init['rot6d'], dtype=torch.float32, device=device).clone()
        transl = torch.as_tensor(init['transl'], dtype=torch.float32, device=device).clone()
    rot6d = rot6d.clone().requires_grad_(True)
    transl = transl.clone().requires_grad_(True)

    shape_fixed = fixed_shape_logit is not None
    if shape_fixed:
        shape_logit = torch.as_tensor(fixed_shape_logit, dtype=torch.float32, device=device).clone()
    else:
        shape_logit = torch.zeros(11, device=device, requires_grad=True)

    labels = list(anny_model.phenotype_labels)

    def forward():
        R = roma.special_gramschmidt(rot6d.reshape(-1, 3, 2)).reshape(1, n_bones, 3, 3)
        homo = rotation_to_homogeneous(R)
        shp = torch.sigmoid(shape_logit).unsqueeze(0)
        kw = {k: shp[:, i] for i, k in enumerate(labels)}
        out = anny_model(pose_parameters=homo, phenotype_kwargs=kw)
        j = out['bone_poses'][0, :, :3, -1]        # [n_bones, 3]
        return j + transl, out

    stages = [
        ("root",        [transl, rot6d], iters[0], True),
        ("root+pose",   [transl, rot6d], iters[1], False),
    ]
    if not shape_fixed:
        stages.append(("root+pose+shape", [transl, rot6d, shape_logit], iters[2], False))

    def pose_prior(R):
        # rotation angle^2 per bone via trace: cos(a) = (tr R - 1)/2
        tr = R.reshape(n_bones, 3, 3).diagonal(dim1=-2, dim2=-1).sum(-1)
        ang = torch.acos(torch.clamp((tr - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6))
        return ((ang ** 2) * pose_prior_mask[:, 0]).sum()

    def errors():
        d = torch.sqrt(((forward()[0][anny_idx] - tgt_sel) ** 2).sum(-1))       # [M] m
        return float(d.mean()) * 1000.0, float(d[reliable].mean()) * 1000.0

    err = limb_err = float('inf')
    for name, params, n_it, root_only in stages:
        if n_it <= 0:
            continue
        opt = torch.optim.Adam(params, lr=lr)
        for it in range(n_it):
            opt.zero_grad()
            j_all, _ = forward()
            sq = ((j_all[anny_idx] - tgt_sel) ** 2).sum(-1)                     # [M]
            loss = (w * sq).sum() / w.sum()
            if not root_only:
                R_cur = roma.special_gramschmidt(rot6d.reshape(-1, 3, 2))
                loss = loss + pose_prior(R_cur)          # weights already inside
            loss.backward()
            if rot6d.grad is not None:
                if root_only:
                    m = torch.zeros_like(rot6d.grad); m[0] = 1.0
                    rot6d.grad = rot6d.grad * m
                else:
                    rot6d.grad = rot6d.grad * bone_mask
            if shape_logit.grad is not None:
                shape_logit.grad = shape_logit.grad * shape_mask
            opt.step()
        err, limb_err = errors()
        if verbose:
            print(f"    stage {name:<16} joint err all = {err:.1f} mm   "
                  f"reliable-joints = {limb_err:.1f} mm")

    with torch.no_grad():
        R = roma.special_gramschmidt(rot6d.reshape(-1, 3, 2)).reshape(n_bones, 3, 3)
        shp = torch.sigmoid(shape_logit)
        j_fit_all = forward()[0]                              # [n_bones, 3]
    shp_np = shp.cpu().numpy()
    # Fitted Anny joints gathered into SMPL's 24-joint order (NaN = unmapped),
    # saved alongside the SMPL targets so check_anny_fit.py can overlay them.
    j_fit_smpl_order = np.full((len(SMPL_JOINT_NAMES), 3), np.nan, dtype=np.float32)
    j_fit_smpl_order[smpl_idx] = j_fit_all[anny_idx].cpu().numpy()
    return {
        'anny_rotmat': R.cpu().numpy(),
        'anny_shape': shp_np,
        'anny_shape_named': dict(zip(labels, shp_np.tolist())),
        'transl': transl.detach().cpu().numpy(),
        'joint_err_mm': err,
        'limb_err_mm': limb_err,
        'anny_joints': j_fit_smpl_order,
        'smpl_joints': np.asarray(target_joints, dtype=np.float32),
        # raw optimiser state, for warm-starting the next pass / next frame
        '_rot6d': rot6d.detach().cpu().numpy(),
        '_shape_logit': shape_logit.detach().cpu().numpy(),
    }


def find_valid_frames(seq, T, max_frames, frame_stride, verbose=True):
    """Return up to max_frames frame indices that are safe to fit.

    3DPW marks per-frame validity in 'campose_valid' (occlusion / failed
    tracking produce garbage or NaN pose values). Filter on that flag AND on
    an explicit NaN check of the pose array itself, since not every 3DPW
    variant's validity flag catches every bad frame. Applying the requested
    stride WITHIN the valid set (not on the raw frame range) is what makes
    this robust to invalid stretches at the start of a sequence.
    """
    poses = np.asarray(seq['poses'][0])
    trans = np.asarray(seq['trans'][0]) if 'trans' in seq else np.zeros((T, 3))

    valid = np.ones(T, dtype=bool)
    if 'campose_valid' in seq:
        cv = np.asarray(seq['campose_valid'][0]).astype(bool)
        if len(cv) == T:
            valid &= cv
        else:
            print(f"  WARNING: campose_valid length {len(cv)} != poses length {T}, "
                  f"ignoring it (falling back to NaN-only filtering)")

    nan_pose = np.isnan(poses).any(axis=1)
    nan_trans = np.isnan(trans).any(axis=1) if trans.shape[0] == T else np.zeros(T, bool)
    valid &= ~nan_pose & ~nan_trans

    n_valid = int(valid.sum())
    if verbose:
        print(f"  {n_valid}/{T} frames valid "
              f"(campose_valid + NaN filter); {T - n_valid} excluded")
    if n_valid == 0:
        return []

    valid_idx = np.where(valid)[0]
    picked = valid_idx[::frame_stride][:max_frames]
    return picked.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list_bones', action='store_true',
                    help='print all Anny bone names and exit')
    ap.add_argument('--check_mapping', action='store_true',
                    help='print the resolved SMPL->Anny mapping and exit')
    ap.add_argument('--pkl', type=str, default='',
                    help='a 3DPW sequenceFiles .pkl with SMPL ground truth')
    ap.add_argument('--out', type=str, default='anny_fit.npz')
    ap.add_argument('--max_frames', type=int, default=4)
    ap.add_argument('--frame_stride', type=int, default=50,
                    help='3DPW is video; skip frames (within the VALID set) so '
                         'the fitted set is varied')
    ap.add_argument('--smpl_model_path', type=str, default='',
                    help='path to SMPL_NEUTRAL.pkl (needed only with --pkl)')
    ap.add_argument('--iters', type=int, nargs=3, default=[150, 400, 250],
                    help='iterations for stage 1 / 2 / 3')
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--no_shared_shape', action='store_true',
                    help='skip pass 2 (one shape per person); keep per-frame shape')
    ap.add_argument('--refine_iters', type=int, default=200,
                    help='pose-only iterations in pass 2 (shape frozen)')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    anny_model = build_anny().to(device)
    labels = list(anny_model.bone_labels)
    pheno_labels = list(anny_model.phenotype_labels)
    print(f"Anny has {len(labels)} bones, {len(pheno_labels)} phenotypes: {pheno_labels}")

    if args.list_bones:
        print("\nAll Anny bone names:")
        for i, n in enumerate(labels):
            print(f"  [{i:3d}] {n}")
        return

    pairs = resolve_mapping(labels)
    if args.check_mapping:
        return
    if len(pairs) < 8:
        print("\nABORT: too few joints mapped to fit anything meaningful.")
        print("Run --list_bones, then fix SMPL_TO_ANNY at the top of this file.")
        sys.exit(1)

    smpl_idx = [p[0] for p in pairs]
    anny_idx = [p[1] for p in pairs]

    if not args.pkl:
        print("\nNo --pkl given; mapping check only. Supply a 3DPW .pkl to fit.")
        return

    if not os.path.isfile(args.pkl):
        print(f"ERROR: not found: {args.pkl}"); sys.exit(1)
    with open(args.pkl, 'rb') as f:
        seq = pickle.load(f, encoding='latin1')
    print(f"\nloaded {args.pkl}")
    print(f"  keys: {list(seq.keys())[:12]}")

    poses = np.asarray(seq['poses'][0])          # [T, 72]
    betas = np.asarray(seq['betas'][0])[:10]     # [10]
    T = poses.shape[0]
    print(f"  {T} frames total")

    frames = find_valid_frames(seq, T, args.max_frames, args.frame_stride)
    if not frames:
        print("\nABORT: no valid frames found in this sequence (all NaN / "
              "campose_valid=False). Try a different .pkl.")
        sys.exit(1)
    print(f"  fitting frames: {frames}")

    import smplx
    if not args.smpl_model_path:
        print("ERROR: --smpl_model_path is required (path to SMPL_NEUTRAL.pkl)")
        sys.exit(1)
    smpl = smplx.create(model_path=args.smpl_model_path, model_type='smpl',
                        gender='neutral', batch_size=1).to(device)

    # ---------------- PASS 1: per frame, shape free ----------------
    # Warm-start each frame from the previous fitted frame (3DPW is video, so
    # neighbouring picked frames are similar poses) - faster & fewer local minima.
    results = []
    prev_init = None
    for k, t in enumerate(frames):
        p = torch.tensor(poses[t], dtype=torch.float32, device=device).reshape(1, 72)
        b = torch.tensor(betas, dtype=torch.float32, device=device).reshape(1, 10)
        with torch.no_grad():
            so = smpl(global_orient=p[:, :3], body_pose=p[:, 3:],
                      betas=b, transl=torch.zeros(1, 3, device=device))
        j_smpl = so.joints[0, :24].cpu().numpy()

        if np.isnan(j_smpl).any():
            print(f"\n[frame {t}] SMPL forward produced NaN despite the valid-frame "
                  f"filter - skipping this frame.")
            continue

        print(f"\n[frame {t}] ({k+1}/{len(frames)}) pass 1 (shape free)...")
        r = fit_frame(anny_model, j_smpl, anny_idx, smpl_idx, device,
                      iters=tuple(args.iters), lr=args.lr, verbose=True, init=prev_init)
        # Never let a NaN fit into the output file: a saved .npz full of NaN
        # looks like valid training data until it silently poisons a run.
        if not np.isfinite(r['joint_err_mm']) or not np.isfinite(r['anny_rotmat']).all():
            print(f"  -> frame {t} produced NaN/Inf - DISCARDED, not saved.")
            prev_init = None
            continue
        r['frame'] = t
        results.append(r)
        prev_init = {'rot6d': r['_rot6d'], 'transl': r['transl']}
        print(f"  -> joint err all {r['joint_err_mm']:.1f} mm | reliable {r['limb_err_mm']:.1f} mm")
        print(f"  -> fitted phenotypes: "
              + "  ".join(f"{k2}={v2:.2f}" for k2, v2 in r['anny_shape_named'].items()))

    if not results:
        print("\nABORT: every candidate frame failed. Try a different sequence "
              "or check the SMPL model file.")
        sys.exit(1)

    # ---------------- PASS 2: ONE shape per person, pose re-fit ----------------
    # 3DPW: one .pkl = one sequence, seq['poses'][0] = one person, so all frames
    # here share a body. Take the median of the per-frame shape logits (robust
    # to the odd bad frame), freeze it, and re-optimise pose only from the pass-1
    # solution. Output phenotypes are then identical across frames, as they
    # should be for a training label.
    if not args.no_shared_shape:
        shared_logit = np.median(np.stack([r['_shape_logit'] for r in results]), axis=0)
        shared_shape = 1.0 / (1.0 + np.exp(-shared_logit))
        print("\nPASS 2: shared shape for this person: "
              + "  ".join(f"{k2}={v2:.2f}" for k2, v2 in zip(pheno_labels, shared_shape)))
        for r in results:
            t = r['frame']
            r2 = fit_frame(anny_model, r['smpl_joints'], anny_idx, smpl_idx, device,
                           iters=(0, args.refine_iters, 0), lr=args.lr, verbose=False,
                           fixed_shape_logit=shared_logit,
                           init={'rot6d': r['_rot6d'], 'transl': r['transl']})
            if not np.isfinite(r2['joint_err_mm']) or not np.isfinite(r2['anny_rotmat']).all():
                print(f"  [frame {t}] pass 2 produced NaN - keeping pass-1 result")
                continue
            print(f"  [frame {t}] all {r['joint_err_mm']:.1f} -> {r2['joint_err_mm']:.1f} mm | "
                  f"reliable {r['limb_err_mm']:.1f} -> {r2['limb_err_mm']:.1f} mm")
            r2['frame'] = t
            r.update(r2)

    errs = [r['joint_err_mm'] for r in results]
    lerrs = [r['limb_err_mm'] for r in results]
    print(f"\nfitted {len(results)} frames; joint error (all 24)  "
          f"mean {np.mean(errs):.1f} mm, max {np.max(errs):.1f} mm")
    print(f"                   reliable joints only  "
          f"mean {np.mean(lerrs):.1f} mm, max {np.max(lerrs):.1f} mm")
    print("INTERPRETATION: judge by the RELIABLE-joint error. <20mm good, >60mm")
    print("means the mapping or the optimisation is wrong - do NOT train on those.")
    print("The all-24 number includes spine1/spine3/hips whose definitions differ")
    print("between the rigs and can never reach zero.")

    np.savez(args.out,
             anny_rotmat=np.stack([r['anny_rotmat'] for r in results]),
             anny_shape=np.stack([r['anny_shape'] for r in results]),
             transl=np.stack([r['transl'] for r in results]),
             joint_err_mm=np.array(errs),
             limb_err_mm=np.array(lerrs),
             joint_weights=np.array([JOINT_WEIGHTS.get(n, 1.0) for n in SMPL_JOINT_NAMES]),
             frames=np.array([r['frame'] for r in results]),
             smpl_idx=np.array(smpl_idx), anny_idx=np.array(anny_idx),
             # --- added for check_anny_fit.py ---
             smpl_joints=np.stack([r['smpl_joints'] for r in results]),   # [F,24,3] m
             anny_joints=np.stack([r['anny_joints'] for r in results]),   # [F,24,3] m
             phenotype_labels=np.array(pheno_labels),
             bone_labels=np.array(labels),
             sequence=os.path.basename(args.pkl))
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()