"""
SMPL -> Anny parameter conversion for 3DPW.   (v2.1 - after job 3455031)

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
    1. Take the 24 SMPL joints of the frame (3DPW's own 'jointPositions', or
       an SMPL forward pass - see TARGET below)
    2. Optimise Anny's pose + shape so Anny's joints land on SMPL's joints
    3. Report the residual error (total AND per joint) so bad fits and bad
       correspondences can be spotted

Optimisation is STAGED, which matters a lot for convergence:
    Stage 1  global orientation + translation only  (align the body as a rigid
             object first - optimising limbs while the root is wrong wastes
             effort and lands in bad local minima)
    Stage 2  + body joint rotations
    Stage 3  + shape phenotypes  (shape last: it is the weakest signal and will
             otherwise absorb pose error by deforming the body)

WHAT CHANGED IN v2.1 AND WHY (job 3455031, outdoors_fencing_01, 38 frames)
--------------------------------------------------------------------------
v2 brought the mean error from 34.3 to 24.6 mm and the limbs/head to 1-10 mm,
but the per-joint table showed the torso still at 43-73 mm (spine1 73, collars
55-59, spine3 58, pelvis 53, spine2 43). That is the size of the DEFINITION gap
between the rigs even in the rest pose: SMPL's spine3 sits 6.5 cm from the
nearest Anny bone head, its collars 6 cm further out. A rigid spine cannot put
spine1 and spine3 on their targets at once, so the optimiser shifted the pelvis
5 cm and drove 'proportions' to 0.06 to change the torso length - a phenotype
value that is wrong for the person and must not become a training label.

  1. VIRTUAL JOINTS (--offset_joints). The two rest skeletons tell us exactly
     where each SMPL torso joint sits relative to its Anny bone. The fit now
     asks the point  bone_head + R_bone @ offset  to reach the SMPL joint, not
     the bone head itself. The offsets are measured once (rest poses aligned by
     body axes and scaled by pelvis->neck height), expressed in the bone's own
     frame, printed as a table, and saved in the .npz. This is v3's "torso
     joints are defined elsewhere" insight turned into a correction instead of
     a down-weighting, so the all-24 error is honest again.
  2. SHAPE PRIOR (--shape_reg): a weak pull of the free phenotypes towards 0.5
     so one remaining joint gap can no longer drag a phenotype to an extreme.
  3. BONE MASK (--free_bones mapped, from v3): only the root and the bones that
     carry a mapped joint may rotate. The other ~130 (twist bones, fingers,
     toes, spine03/05, neck02/03, face) have no target and drifted in v1.
  4. WARM START (from v3): pass 1 now fits EVERY frame, each starting from the
     previous one (3DPW is video), with fewer iterations (--warm_iters). Pass 2
     freezes the median body and refits pose only, from the pass-1 solution
     (--refine_iters). v2 fitted 8 frames in pass 1 and refitted all 40 cold in
     pass 2 at 24 s/frame; this should be several times faster per frame.

WHAT CHANGED IN v2 AND WHY (job 3454323)
----------------------------------------
Job 3454323 fitted 140 frames of 4 scenes to a ~34 mm floor that was the SAME
for every frame of every scene, and all four (different!) people came out with
the SAME body: gender~0.02, age=0.67, height~0.33, asian~0.73. A floor that
does not depend on the pose, plus a body that does not depend on the person,
means the shape was absorbing a fixed skeleton-definition mismatch, not
describing anyone. v2 attacks that from several sides:

  1. TARGET = 3DPW's own 'jointPositions' (--target gt, default). v1 ran the
     NEUTRAL SMPL model with the sequence's betas, but 3DPW betas belong to the
     GENDERED models; plugging them into the neutral model gives a different
     skeleton. 3DPW ships the correct world-space joints per frame, so use them
     (minus 'trans', to stay in the transl=0 model frame that the renderer
     expects). If the SMPL model path is still given, the script prints how far
     the neutral-model joints were from the GT for the first frame.

  2. MAPPING CHECKED BY GEOMETRY, not only by name. MakeHuman (Anny's rig)
     numbers the spine TOP-DOWN: spine01 is the bone just below the neck,
     spine05 sits on the pelvis. SMPL numbers BOTTOM-UP: spine1 is lumbar.
     v1 mapped spine1->spine01 etc. Likewise 'pelvis.L' is a short bone from
     the root out to the hip; its HEAD is at the pelvis centre, the hip joint
     itself is the head of 'upperleg01.L'. v1 put SMPL's hips on pelvis.L/R,
     i.e. three targets on one point. Rather than trust another guess, v2
     computes both rest skeletons and (a) prints them side by side
     (--check_mapping), (b) picks the spine bones by height and the hip bones
     by distance from the pelvis (--auto_mapping 1, default, needs the SMPL
     model). The per-joint error table at the end shows whether anything is
     still off: a correct fit has a flat table, a wrong correspondence sticks
     out by a factor of 3-10.

  3. SHAPE: 3DPW tells us the gender, so gender is FIXED from the .pkl
     (--fix_gender 1), and only the phenotypes that move the skeleton are fit
     (--free_phenotypes age,height,proportions). Joints carry no information
     about weight, muscle, cup size, firmness or ethnicity; letting the
     optimiser move them only lets it hide skeleton error in them (v1: weight
     0.02..0.65 across frames of one person). They stay at the neutral 0.5.

  4. POSE REGULARISER (--pose_reg): 163 free rotations vs 24 target points is
     badly under-determined; twist about a bone's own axis moves no joint but
     visibly corkscrews the mesh, and the extra MakeHuman bones (upperleg02,
     lowerleg02, spine04/05, neck02/03, fingers, face) are unconstrained. A
     tiny pull of every non-root rotation towards identity settles the
     null space (with Adam even a tiny weight is enough) and costs ~1 mm on
     the real joints. --pose_reg 0 reproduces v1.

  5. TRANSLATION INIT: Anny's root starts on the target pelvis instead of at
     the origin (stage 1 then only has to find the rotation).

  6. OUTPUT: the .npz now also holds the target joints, the fitted Anny joints
     and per-joint errors, so render_anny_fit.py can draw both on the photo,
     and 'img_frame_ids' now holds the IMAGE numbers (= 'frames'). The v1
     field was 3DPW's 60 Hz index map (0,2,4,...), which is what made the v1
     renderer draw pose frame t on image 2t.

THE JOINT CORRESPONDENCE PROBLEM
--------------------------------
SMPL has 24 joints; Anny has 163 bones with different names and a different
rest pose. The fit is only as good as the mapping between them. Anny's
'bone_poses' give each bone's HEAD (its own origin), so an SMPL joint must be
mapped to the Anny bone whose head sits at that joint, not to the bone that
ends there.

THE 11 PHENOTYPES: NOT MAPPED - PARTLY FIXED, PARTLY OPTIMISED
---------------------------------------------------------------
There is no SMPL equivalent of Anny's phenotypes to map from. SMPL's shape
space is 10 abstract PCA "betas"; Anny's 11 phenotypes (gender, age, muscle,
weight, height, proportions, cupsize, firmness, african, asian, caucasian) are
semantic. Gender comes from 3DPW; age/height/proportions are fit to the
joints; the rest stay neutral. (MakeHuman convention, which Anny inherits:
gender 0 = female, 1 = male; age 0.5 = 25 y, 1.0 = 90 y.)

THE NaN BUG (job 3409031) AND ITS FIX
--------------------------------------
Every frame came back NaN from iteration 1 of stage 1: the 6D rotation was
initialised with a degenerate pair of columns. Fixed in fit_frame (indices 0
and 3). 3DPW also has invalid frames ('campose_valid' false / NaN values);
find_valid_frames() filters those before the stride is applied.

USAGE
    python smpl_to_anny.py --list_bones                              # inspect names
    python smpl_to_anny.py --check_mapping --smpl_model_path SMPL_NEUTRAL.pkl
                                                                     # rest skeletons side by side
    python smpl_to_anny.py --pkl <3dpw.pkl> --smpl_model_path <SMPL_NEUTRAL.pkl> \\
                           --out fit.npz --max_frames 40 --frame_stride 25
"""

import argparse
import os
import pickle
import sys
import warnings

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
SMPL_IDX = {n: i for i, n in enumerate(SMPL_JOINT_NAMES)}

# ---------------------------------------------------------------------------
# SMPL joint -> candidate Anny bone names, in priority order (first that
# exists wins). Bone NAMES verified against the real 163-bone list (job
# 3409000); the spine/hip choices are additionally checked by GEOMETRY at run
# time when the SMPL model is available (see refine_mapping_by_geometry).
# ---------------------------------------------------------------------------
SMPL_TO_ANNY = {
    'pelvis':         ['root', 'pelvis', 'hips'],
    'left_hip':       ['upperleg01.L', 'pelvis.L', 'thigh.L', 'hip.L'],
    'right_hip':      ['upperleg01.R', 'pelvis.R', 'thigh.R', 'hip.R'],
    'spine1':         ['spine04', 'spine05', 'spine1', 'spine_01'],
    'left_knee':      ['lowerleg01.L', 'shin.L', 'lowerleg.L', 'knee.L'],
    'right_knee':     ['lowerleg01.R', 'shin.R', 'lowerleg.R', 'knee.R'],
    'spine2':         ['spine03', 'spine2', 'spine_02'],
    'left_ankle':     ['foot.L', 'ankle.L'],
    'right_ankle':    ['foot.R', 'ankle.R'],
    'spine3':         ['spine02', 'spine01', 'spine3', 'spine_03', 'chest'],
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

# Phenotypes that actually move the skeleton. Everything else (muscle, weight,
# cupsize, firmness, african/asian/caucasian) changes the surface, not the
# joints, so the joint fit cannot determine it - it stays neutral.
DEFAULT_FREE_PHENOTYPES = 'age,height,proportions'


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


# ---------------------------------------------------------------------------
# Rest-pose geometry: used to check / refine the mapping
# ---------------------------------------------------------------------------
def anny_rest_positions(anny_model, device):
    """Head position of every Anny bone in the rest pose, neutral (0.5)
    phenotypes. [n_bones, 3] numpy, Anny's own frame."""
    from utils import rotation_to_homogeneous
    n_bones = len(anny_model.bone_labels)
    labels = list(anny_model.phenotype_labels)
    R = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(1, n_bones, 1, 1)
    shp = torch.full((1, len(labels)), 0.5, device=device)
    kw = {k: shp[:, i] for i, k in enumerate(labels)}
    with torch.no_grad():
        out = anny_model(pose_parameters=rotation_to_homogeneous(R), phenotype_kwargs=kw)
    return out['bone_poses'][0, :, :3, -1].cpu().numpy()


def anny_rest_frames(anny_model, device):
    """Rotation part of every bone's transform in the rest pose. [n_bones, 3, 3]
    numpy. Needed to express a joint offset in the bone's own frame."""
    from utils import rotation_to_homogeneous
    n_bones = len(anny_model.bone_labels)
    labels = list(anny_model.phenotype_labels)
    R = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(1, n_bones, 1, 1)
    shp = torch.full((1, len(labels)), 0.5, device=device)
    kw = {k: shp[:, i] for i, k in enumerate(labels)}
    with torch.no_grad():
        out = anny_model(pose_parameters=rotation_to_homogeneous(R), phenotype_kwargs=kw)
    return out['bone_poses'][0, :, :3, :3].cpu().numpy()


def smpl_rest_joints(smpl, device):
    """The 24 SMPL joints in the zero pose with zero betas. [24, 3] numpy."""
    with torch.no_grad():
        so = smpl(global_orient=torch.zeros(1, 3, device=device),
                  body_pose=torch.zeros(1, 69, device=device),
                  betas=torch.zeros(1, 10, device=device),
                  transl=torch.zeros(1, 3, device=device))
    return so.joints[0, :24].cpu().numpy()


def _up_axis(head_offset):
    """Which axis is 'up' and its sign, from the pelvis->head vector."""
    a = int(np.argmax(np.abs(head_offset)))
    s = 1.0 if head_offset[a] >= 0 else -1.0
    return a, s


def print_rest_table(pairs, bone_labels, ja, js):
    """Both rest skeletons side by side, offsets from their own pelvis/root."""
    by_name = {p[2]: p for p in pairs}
    root_i = by_name['pelvis'][1] if 'pelvis' in by_name else 0
    ja_off = ja - ja[root_i]
    print("\nREST-POSE CHECK  (offsets from own pelvis/root, metres; |d| = distance from pelvis)")
    if js is None:
        print("  (no --smpl_model_path given: only the Anny side is shown)")
    hdr = f"{'SMPL joint':<15}"
    if js is not None:
        hdr += f" {'SMPL offset (x y z)':>26} {'|d|':>6} |"
    hdr += f" {'Anny bone':<15} {'Anny offset (x y z)':>26} {'|d|':>6}"
    print(hdr)
    print("-" * len(hdr))
    js_off = js - js[0] if js is not None else None
    for si, ai, sn, an in pairs:
        line = f"{sn:<15}"
        if js_off is not None:
            o = js_off[si]
            line += f" {o[0]:8.3f} {o[1]:8.3f} {o[2]:8.3f} {np.linalg.norm(o):6.3f} |"
        o = ja_off[ai]
        line += f" {an:<15} {o[0]:8.3f} {o[1]:8.3f} {o[2]:8.3f} {np.linalg.norm(o):6.3f}"
        print(line)
    print("Torso, hip and knee rows should agree to a few cm. Arm rows WILL differ: SMPL")
    print("rests in a T-pose, Anny in an A-pose. If the 'head' row is tall on a different")
    print("axis on each side, the two rigs simply use different up-axes; compare |d|.")

    # All spine/neck bones of Anny with their height - the spine choice in a glance
    ua, us = _up_axis(ja_off[by_name['head'][1]]) if 'head' in by_name else (1, 1.0)
    print("\nAnny torso bones by height above root (rest pose):")
    for i, n in enumerate(bone_labels):
        nl = n.lower()
        if nl.startswith(('spine', 'neck', 'head', 'root', 'pelvis', 'upperleg01', 'clavicle')):
            print(f"  {n:<16} height {ja_off[i, ua] * us:7.3f} m   |d| {np.linalg.norm(ja_off[i]):6.3f} m")
    if js_off is not None:
        sa, ss = _up_axis(js_off[SMPL_IDX['head']])
        print("SMPL torso joints by height above pelvis (rest pose):")
        for n in ('pelvis', 'left_hip', 'spine1', 'spine2', 'spine3', 'neck', 'head', 'left_collar'):
            i = SMPL_IDX[n]
            print(f"  {n:<16} height {js_off[i, sa] * ss:7.3f} m   |d| {np.linalg.norm(js_off[i]):6.3f} m")


def refine_mapping_by_geometry(pairs, bone_labels, ja, js, verbose=True):
    """Pick the hip bones by distance from the pelvis and the spine bones by
    relative height, using both rest skeletons. Heights are measured from the
    HIP MIDPOINT along hip-midpoint->neck (the same anchor the virtual-joint
    offsets use), because SMPL's pelvis joint sits ~9 cm above its hips while
    Anny's root sits at hip height. Returns a new pairs list; no-op when the
    SMPL rest joints are not available."""
    if js is None:
        if verbose:
            print("\n(auto-mapping skipped: needs --smpl_model_path for the SMPL rest skeleton)")
        return pairs
    lower = {n.lower(): i for i, n in enumerate(bone_labels)}
    by_name = {p[2]: list(p) for p in pairs}
    if not all(k in by_name for k in ('pelvis', 'neck', 'left_hip', 'right_hip')):
        return pairs
    root_i = by_name['pelvis'][1]
    changed = []

    # --- hips first: the bone whose head is as far from the pelvis as SMPL's hip joint
    for sname, side in (('left_hip', 'L'), ('right_hip', 'R')):
        si = SMPL_IDX[sname]
        cands = [(n, lower[n.lower()]) for n in (f'upperleg01.{side}', f'pelvis.{side}') if n.lower() in lower]
        if len(cands) < 2:
            continue
        target = float(np.linalg.norm(js[si] - js[0]))
        n, bi = min(cands, key=lambda c: abs(float(np.linalg.norm(ja[c[1]] - ja[root_i])) - target))
        old = by_name[sname][3] if sname in by_name else None
        by_name[sname] = [si, bi, sname, n]
        if old != n:
            changed.append(f"{sname}: {old} -> {n} (SMPL hip is {target * 100:.1f} cm from the pelvis; "
                           f"{n} head is {np.linalg.norm(ja[bi] - ja[root_i]) * 100:.1f} cm from root)")

    # --- spine: heights from the hip midpoint, normalised by hip-midpoint->neck
    o_a = 0.5 * (ja[by_name['left_hip'][1]] + ja[by_name['right_hip'][1]])
    o_s = 0.5 * (js[SMPL_IDX['left_hip']] + js[SMPL_IDX['right_hip']])
    up_a = ja[by_name['neck'][1]] - o_a
    up_s = js[SMPL_IDX['neck']] - o_s
    h_a, h_s = float(np.linalg.norm(up_a)), float(np.linalg.norm(up_s))
    if h_a <= 0.05 or h_s <= 0.05:
        if verbose:
            print("\n(auto-mapping skipped: could not establish a hip->neck axis)")
        return [tuple(by_name[n]) for n in SMPL_JOINT_NAMES if n in by_name]
    Ha = (ja - o_a) @ (up_a / h_a) / h_a         # Anny bone heights, 0 = hip midpoint, 1 = neck
    Hs = (js - o_s) @ (up_s / h_s) / h_s         # SMPL joint heights, same scale
    from itertools import combinations
    spine_bones = [(n, lower[n]) for n in ('spine01', 'spine02', 'spine03', 'spine04', 'spine05') if n in lower]
    if len(spine_bones) >= 3:
        spine_bones.sort(key=lambda nb: Ha[nb[1]])           # by height, lowest first
        smpl_spine = [('spine1', SMPL_IDX['spine1']), ('spine2', SMPL_IDX['spine2']), ('spine3', SMPL_IDX['spine3'])]
        best = min(combinations(spine_bones, 3),
                   key=lambda trip: sum(abs(Ha[bi] - Hs[si]) for (n, bi), (sn, si) in zip(trip, smpl_spine)))
        for (n, bi), (sname, si) in zip(best, smpl_spine):
            old = by_name[sname][3] if sname in by_name else None
            by_name[sname] = [si, bi, sname, n]
            if old != n:
                msg = f"{sname}: {old} -> {n} (SMPL {sname} sits at {Hs[si]:.2f} of hip->neck, {n} at {Ha[bi]:.2f}"
                if old and old.lower() in lower:
                    msg += f", {old} at {Ha[lower[old.lower()]]:.2f}"
                changed.append(msg + ")")
    if verbose:
        if changed:
            print("\nAUTO-MAPPING (rest-pose geometry) changed these correspondences:")
            for c in changed:
                print("  " + c)
        else:
            print("\nAUTO-MAPPING: name-based spine/hip choices agree with the rest-pose geometry.")
    return [tuple(by_name[n]) for n in SMPL_JOINT_NAMES if n in by_name]


# ---------------------------------------------------------------------------
# Virtual joints: close the torso definition gap between the two rigs
# ---------------------------------------------------------------------------
def _body_basis(origin, top, lhip, rhip):
    """Right-handed body frame from rest-pose points: columns = lateral
    (right->left), forward, up, with up = origin->top. Works for any rig
    regardless of its axes."""
    up = top - origin
    up = up / (np.linalg.norm(up) + 1e-9)
    lat = lhip - rhip
    lat = lat - (lat @ up) * up
    lat = lat / (np.linalg.norm(lat) + 1e-9)
    fwd = np.cross(lat, up)
    return np.stack([lat, fwd, up], axis=1)


def compute_joint_offsets(pairs, bone_labels, ja, Ra, js, offset_joints, verbose=True):
    """For each SMPL joint in `offset_joints`, the vector from the head of its
    Anny bone to where the SMPL joint sits, in the BONE's own frame, measured on
    the two rest skeletons after aligning their body axes and scale.

    ANCHOR: the two skeletons are aligned at the HIP MIDPOINT (mean of the two
    hip joints) with 'up' along hip-midpoint -> neck, not at the pelvis/root.
    SMPL's pelvis joint sits ~9 cm above its hip joints while Anny's root sits
    at hip height and ~7 cm in front of them, so anchoring at the pelvis would
    push that whole discrepancy into the hip and leg offsets. The hip joints
    and the spine axis are the closest thing to a shared anatomical reference.

    At fit time the point  head + R_bone @ offset  is what has to reach the SMPL
    joint, instead of the bone head itself. Returns {smpl_idx: offset[3]} and
    prints a table. Empty dict (= v2 behaviour) when the SMPL rest joints are
    not available."""
    if js is None:
        if verbose:
            print("\n(joint offsets skipped: need --smpl_model_path; torso joints will be "
                  "fitted to the bone heads as in v2)")
        return {}
    by_name = {p[2]: p for p in pairs}
    need = ('pelvis', 'neck', 'left_hip', 'right_hip')
    if not all(n in by_name for n in need):
        return {}
    a_l, a_r, a_n = by_name['left_hip'][1], by_name['right_hip'][1], by_name['neck'][1]
    o_a = 0.5 * (ja[a_l] + ja[a_r])                       # Anny hip midpoint
    o_s = 0.5 * (js[SMPL_IDX['left_hip']] + js[SMPL_IDX['right_hip']])
    Ba = _body_basis(o_a, ja[a_n], ja[a_l], ja[a_r])
    Bs = _body_basis(o_s, js[SMPL_IDX['neck']], js[SMPL_IDX['left_hip']], js[SMPL_IDX['right_hip']])
    R_s2a = Ba @ Bs.T                                     # SMPL axes -> Anny axes
    h_a = float(np.linalg.norm(ja[a_n] - o_a))
    h_s = float(np.linalg.norm(js[SMPL_IDX['neck']] - o_s))
    scale = h_a / h_s if h_s > 0.05 else 1.0
    offsets, rows = {}, []
    for name in offset_joints:
        if name not in by_name:
            continue
        si, ai = by_name[name][0], by_name[name][1]
        target_a = o_a + scale * (R_s2a @ (js[si] - o_s))   # SMPL joint placed in Anny's rest frame
        d_world = target_a - ja[ai]
        d_local = Ra[ai].T @ d_world
        offsets[si] = d_local.astype(np.float32)
        rows.append((name, by_name[name][3], d_world, float(np.linalg.norm(d_world))))
    if verbose:
        print(f"\nVIRTUAL-JOINT OFFSETS (rest pose, anchored at the hip midpoint; SMPL rest scaled "
              f"x{scale:.3f} to Anny's hip->neck length). The fit moves the point head+offset onto "
              f"the SMPL joint:")
        print(f"  {'SMPL joint':<14} {'Anny bone':<14} {'offset in Anny frame (x y z) cm':>34} {'|d| cm':>7}")
        for name, bone, d, n in rows:
            print(f"  {name:<14} {bone:<14} {d[0]*100:10.1f} {d[1]*100:7.1f} {d[2]*100:7.1f} {n*100:10.1f}")
        big = [r for r in rows if r[3] > 0.15]
        if big:
            print("  WARNING: offsets above 15 cm usually mean a wrong correspondence, not a "
                  "definition gap: " + ", ".join(r[0] for r in big))
    return offsets



# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------
def fit_frame(anny_model, target_joints, anny_idx, smpl_idx, device,
              iters=(150, 400, 250), lr=0.05, verbose=False, fixed_shape=None,
              shape_init=None, free_shape=None, pose_reg=1e-4, shape_reg=1e-3,
              offsets=None, free_bones='mapped', init=None):
    """Optimise Anny pose (+ shape) so its joints match `target_joints`.

    target_joints: [24, 3] SMPL joint positions in metres (model frame)
    anny_idx / smpl_idx: the mapped bone / joint indices (same length)
    fixed_shape: [n_pheno] -> all phenotypes frozen at these values (pass 2)
    shape_init:  [n_pheno] starting values (default 0.5 everywhere)
    free_shape:  [n_pheno] bool - which phenotypes may move (default: all)
    pose_reg:    pull of every optimised non-root rotation towards identity
    shape_reg:   pull of the free phenotypes towards 0.5
    offsets:     {smpl_idx: [3]} virtual-joint offsets in the bone frame (see
                 compute_joint_offsets); joints not listed use the bone head
    free_bones:  'mapped' = only the root and bones carrying a mapped joint may
                 rotate (the other ~130 are unconstrained and would drift);
                 'all' = v2 behaviour
    init:        {'rot6d','transl'[,'shape_logit']} to warm-start from (previous
                 frame, or this frame's pass-1 solution)

    Returns dict with anny_rotmat (n_bones,3,3), anny_shape (n_pheno,),
    transl (3,), joint_err_mm, per_joint_err_mm (24, NaN = unmapped),
    anny_joints (24,3 fitted virtual-joint positions incl. transl), the fitted
    phenotype values by name, and the raw optimiser state for warm-starting.
    """
    from utils import rotation_to_homogeneous
    import roma

    n_bones = len(anny_model.bone_labels)
    labels = list(anny_model.phenotype_labels)
    n_pheno = len(labels)
    M = len(anny_idx)
    tgt = torch.as_tensor(np.asarray(target_joints), dtype=torch.float32, device=device)
    tgt_sel = tgt[smpl_idx]                       # [M, 3]
    root_bone = anny_idx[smpl_idx.index(0)] if 0 in smpl_idx else 0
    anny_idx_t = torch.as_tensor(anny_idx, dtype=torch.long, device=device)

    # virtual-joint offsets, one row per mapped joint (zero = bone head)
    off = torch.zeros(M, 3, device=device)
    if offsets:
        for k, si in enumerate(smpl_idx):
            if si in offsets:
                off[k] = torch.as_tensor(offsets[si], dtype=torch.float32, device=device)

    # which bones may rotate
    bone_mask = torch.zeros(n_bones, 1, device=device)
    if free_bones == 'all':
        bone_mask[:] = 1.0
    else:
        bone_mask[root_bone] = 1.0
        bone_mask[anny_idx_t] = 1.0

    # 6D IDENTITY ROTATION - indices 0 and 3, NOT 0 and 4.
    # The 6 numbers reshape to a 3x2 matrix taken COLUMN-wise; indices 0 and 3
    # give columns (1,0,0) and (0,1,0) -> identity. (0 and 4 gave a degenerate
    # matrix whose Gram-Schmidt divides by zero -> the all-NaN fits of jobs
    # 3409031 / 3413616.)
    rot6d = torch.zeros(n_bones, 6, device=device)
    rot6d[:, 0] = 1.0
    rot6d[:, 3] = 1.0
    transl = torch.zeros(3, device=device)
    warm = init is not None and 'rot6d' in init and 'transl' in init
    if warm:
        rot6d = torch.as_tensor(np.asarray(init['rot6d']), dtype=torch.float32, device=device)
        transl = torch.as_tensor(np.asarray(init['transl']), dtype=torch.float32, device=device)
    rot6d = rot6d.clone().requires_grad_(True)
    transl = transl.clone().requires_grad_(True)

    # SHAPE. Parametrised as logits so sigmoid keeps every phenotype in (0,1).
    if fixed_shape is not None:
        base = torch.as_tensor(np.asarray(fixed_shape), dtype=torch.float32, device=device)
        base = base.clamp(1e-4, 1 - 1e-4)
        shape_logit = torch.log(base / (1 - base))
        shape_logit.requires_grad_(False)
        shape_mask = torch.zeros(n_pheno, device=device)
    else:
        if warm and init.get('shape_logit') is not None:
            shape_logit = torch.as_tensor(np.asarray(init['shape_logit']), dtype=torch.float32, device=device)
        else:
            if shape_init is None:
                base = torch.full((n_pheno,), 0.5, device=device)
            else:
                base = torch.as_tensor(np.asarray(shape_init), dtype=torch.float32, device=device)
            base = base.clamp(1e-4, 1 - 1e-4)
            shape_logit = torch.log(base / (1 - base))
        shape_logit = shape_logit.clone().requires_grad_(True)
        if free_shape is None:
            shape_mask = torch.ones(n_pheno, device=device)
        else:
            shape_mask = torch.as_tensor(np.asarray(free_shape), dtype=torch.float32, device=device)
    n_free_shape = float(shape_mask.sum().item())

    eye3 = torch.eye(3, device=device)
    reg_mask = bone_mask[:, 0].clone()
    reg_mask[root_bone] = 0.0                     # global orientation is never regularised

    def forward():
        R = roma.special_gramschmidt(rot6d.reshape(-1, 3, 2)).reshape(1, n_bones, 3, 3)
        homo = rotation_to_homogeneous(R)
        shp = torch.sigmoid(shape_logit).unsqueeze(0)
        kw = {k: shp[:, i] for i, k in enumerate(labels)}
        out = anny_model(pose_parameters=homo, phenotype_kwargs=kw)
        bp = out['bone_poses'][0]                  # [n_bones, 4, 4]
        heads = bp[anny_idx_t, :3, 3]              # [M, 3] bone heads
        Rb = bp[anny_idx_t, :3, :3]                # [M, 3, 3] bone frames
        pts = heads + torch.bmm(Rb, off.unsqueeze(-1)).squeeze(-1)   # virtual joints
        return pts + transl, R

    # Translation init (cold start only): put the mapped pelvis on the target pelvis.
    if not warm:
        with torch.no_grad():
            p0, _ = forward()
            if 0 in smpl_idx:
                k0 = smpl_idx.index(0)
                transl.add_(tgt_sel[k0] - p0[k0])
            else:
                transl.add_(tgt_sel.mean(0) - p0.mean(0))

    def joint_loss(pts):
        return ((pts - tgt_sel) ** 2).sum(-1).mean()

    def reg_loss(R):
        loss = 0.0
        if pose_reg > 0:
            dev = ((R[0] - eye3) ** 2).sum((-1, -2))   # = 4(1 - cos angle) per bone, smooth at 0
            loss = loss + pose_reg * (dev * reg_mask).sum() / max(float(reg_mask.sum().item()), 1.0)
        if shape_reg > 0 and fixed_shape is None and n_free_shape > 0:
            dev = (torch.sigmoid(shape_logit) - 0.5) ** 2
            loss = loss + shape_reg * (dev * shape_mask).sum() / n_free_shape
        return loss

    stages = [
        ("root",        [transl, rot6d], iters[0], True),
        ("root+pose",   [transl, rot6d], iters[1], False),
    ]
    if fixed_shape is None:
        stages.append(("root+pose+shape", [transl, rot6d, shape_logit], iters[2], False))
    else:
        # shape frozen: spend stage-3 budget on more pose refinement instead
        stages.append(("pose (shape fixed)", [transl, rot6d], iters[2], False))

    err = float('inf')
    for name, params, n_it, root_only in stages:
        if n_it <= 0:
            continue
        opt = torch.optim.Adam(params, lr=lr)
        for it in range(n_it):
            opt.zero_grad()
            pts, R = forward()
            loss = joint_loss(pts) + reg_loss(R)
            loss.backward()
            if rot6d.grad is not None:
                if root_only:
                    g_mask = torch.zeros_like(bone_mask)
                    g_mask[root_bone] = 1.0
                else:
                    g_mask = bone_mask
                rot6d.grad = rot6d.grad * g_mask
            if shape_logit.grad is not None:
                shape_logit.grad = shape_logit.grad * shape_mask
            opt.step()
        with torch.no_grad():
            pts, _ = forward()
            err = float(torch.sqrt(((pts - tgt_sel) ** 2).sum(-1)).mean())
        if verbose:
            print(f"    stage {name:<18} mean joint err = {err * 1000:.1f} mm")

    with torch.no_grad():
        pts, R = forward()
        d = torch.sqrt(((pts - tgt_sel) ** 2).sum(-1)).cpu().numpy()   # [M]
        R = R[0].cpu().numpy()
        shp = torch.sigmoid(shape_logit).cpu().numpy()
        j_fit = pts.cpu().numpy()
    per_joint = np.full(len(SMPL_JOINT_NAMES), np.nan, dtype=np.float32)
    per_joint[smpl_idx] = d * 1000.0
    anny_joints = np.full((len(SMPL_JOINT_NAMES), 3), np.nan, dtype=np.float32)
    anny_joints[smpl_idx] = j_fit
    return {
        'anny_rotmat': R,
        'anny_shape': shp,
        'anny_shape_named': dict(zip(labels, shp.tolist())),
        'transl': transl.detach().cpu().numpy(),
        'joint_err_mm': float(d.mean()) * 1000.0,
        'per_joint_err_mm': per_joint,
        'anny_joints': anny_joints,
        'target_joints': np.asarray(target_joints, dtype=np.float32),
        # raw optimiser state for warm-starting the next frame / pass 2
        '_rot6d': rot6d.detach().cpu().numpy(),
        '_shape_logit': shape_logit.detach().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# 3DPW helpers
# ---------------------------------------------------------------------------
def find_valid_frames(seq, person, T, max_frames, frame_stride, verbose=True):
    """Return up to max_frames frame indices that are safe to fit.

    3DPW marks per-frame validity in 'campose_valid' (occlusion / failed
    tracking produce garbage or NaN pose values). Filter on that flag AND on
    an explicit NaN check of the pose / translation / joint arrays, since not
    every 3DPW variant's validity flag catches every bad frame. The stride is
    applied WITHIN the valid set, so invalid stretches at the start of a
    sequence do not eat the sample.
    """
    poses = np.asarray(seq['poses'][person])
    trans = np.asarray(seq['trans'][person]) if 'trans' in seq else np.zeros((T, 3))

    valid = np.ones(T, dtype=bool)
    if 'campose_valid' in seq:
        cv = np.asarray(seq['campose_valid'][person]).astype(bool)
        if len(cv) == T:
            valid &= cv
        else:
            print(f"  WARNING: campose_valid length {len(cv)} != poses length {T}, "
                  f"ignoring it (falling back to NaN-only filtering)")

    valid &= ~np.isnan(poses).any(axis=1)
    if trans.shape[0] == T:
        valid &= ~np.isnan(trans).any(axis=1)
    if 'jointPositions' in seq:
        jp = np.asarray(seq['jointPositions'][person])
        if jp.shape[0] == T:
            valid &= ~np.isnan(jp.reshape(T, -1)).any(axis=1)

    n_valid = int(valid.sum())
    if verbose:
        print(f"  {n_valid}/{T} frames valid "
              f"(campose_valid + NaN filter); {T - n_valid} excluded")
    if n_valid == 0:
        return []

    valid_idx = np.where(valid)[0]
    picked = valid_idx[::frame_stride][:max_frames]
    return picked.tolist()


def read_gender(seq, person):
    """'m' / 'f' / None from the 3DPW pickle."""
    g = seq.get('genders', seq.get('gender', None))
    if g is None:
        return None
    try:
        g = g[person]
    except (IndexError, TypeError, KeyError):
        return None
    if isinstance(g, bytes):
        g = g.decode('latin1')
    g = str(g).strip().lower()
    if not g:
        return None
    return 'm' if g[0] == 'm' else ('f' if g[0] == 'f' else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list_bones', action='store_true',
                    help='print all Anny bone names and exit')
    ap.add_argument('--check_mapping', action='store_true',
                    help='print the resolved SMPL->Anny mapping, both rest skeletons, and exit')
    ap.add_argument('--pkl', type=str, default='',
                    help='a 3DPW sequenceFiles .pkl with SMPL ground truth')
    ap.add_argument('--person', type=int, default=0,
                    help='subject index inside the .pkl (office_phoneCall_00 has two)')
    ap.add_argument('--out', type=str, default='anny_fit.npz')
    ap.add_argument('--max_frames', type=int, default=4)
    ap.add_argument('--frame_stride', type=int, default=50,
                    help='3DPW is video; skip frames (within the VALID set) so '
                         'the fitted set is varied')
    ap.add_argument('--smpl_model_path', type=str, default='',
                    help='path to SMPL_NEUTRAL.pkl (required for --target smpl; with '
                         '--target gt it is used only for the rest-pose mapping check)')
    ap.add_argument('--target', type=str, default='gt', choices=['gt', 'smpl'],
                    help="gt: 3DPW's own jointPositions (gendered, exact). "
                         "smpl: recompute with the neutral SMPL model (v1 behaviour)")
    ap.add_argument('--auto_mapping', type=int, default=1, choices=[0, 1],
                    help='1: pick spine/hip bones by rest-pose geometry (needs the SMPL model)')
    ap.add_argument('--iters', type=int, nargs=3, default=[150, 400, 250],
                    help='iterations for stage 1 / 2 / 3')
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--pose_reg', type=float, default=1e-4,
                    help='pull of non-root rotations towards identity (0 = off = v1)')
    ap.add_argument('--fix_gender', type=int, default=1, choices=[0, 1],
                    help="1: set Anny's gender phenotype from the 3DPW gender (m->1, f->0)")
    ap.add_argument('--free_phenotypes', type=str, default=DEFAULT_FREE_PHENOTYPES,
                    help="comma list of phenotypes the optimiser may move, or 'all' "
                         "(v1 behaviour). Gender is added automatically if it is not fixed.")
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--shared_shape', type=int, default=1, choices=[0, 1],
                    help='1 (default): two-pass fit. Pass 1 fits shape freely on a few '
                         'frames, the MEDIAN phenotypes become the sequence body, pass 2 '
                         'refits every frame with shape fixed. 0: independent per-frame.')
    ap.add_argument('--shape_frames', type=int, default=0,
                    help='0 (default): the shared body is the median over ALL pass-1 frames; '
                         'N>0: median over N evenly spaced frames only')
    ap.add_argument('--refine_iters', type=int, default=200,
                    help='pass 2: pose-only refit iterations per frame with the shared shape, '
                         'warm-started from the pass-1 solution')
    ap.add_argument('--warm_iters', type=int, nargs=3, default=[30, 150, 100],
                    help='stage iterations for frames warm-started from the previous frame')
    ap.add_argument('--warm_start', type=int, default=1, choices=[0, 1],
                    help='1: start each frame from the previous fitted frame (3DPW is video)')
    ap.add_argument('--shape_reg', type=float, default=1e-3,
                    help='pull of the free phenotypes towards 0.5 (0 = off)')
    ap.add_argument('--free_bones', type=str, default='mapped', choices=['mapped', 'all'],
                    help="mapped: only root + bones carrying a mapped joint may rotate; all = v2")
    ap.add_argument('--offset_joints', type=str,
                    default='pelvis,left_hip,right_hip,spine1,spine2,spine3,neck,left_collar,right_collar',
                    help="SMPL joints fitted as virtual points head+offset (rest-pose measured); "
                         "'' = none (v2 behaviour)")
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

    # SMPL model: needed for --target smpl, optional (but recommended) otherwise.
    smpl = None
    if args.smpl_model_path:
        if not os.path.isfile(args.smpl_model_path):
            print(f"ERROR: SMPL model not found: {args.smpl_model_path}"); sys.exit(1)
        import smplx
        smpl = smplx.create(model_path=args.smpl_model_path, model_type='smpl',
                            gender='neutral', batch_size=1).to(device)

    # Rest-pose geometry: check, and refine, the mapping.
    ja_rest = anny_rest_positions(anny_model, device)
    Ra_rest = anny_rest_frames(anny_model, device)
    js_rest = smpl_rest_joints(smpl, device) if smpl is not None else None
    offset_names = [s.strip() for s in args.offset_joints.split(',') if s.strip()]
    if args.check_mapping:
        print_rest_table(pairs, labels, ja_rest, js_rest)
        pairs2 = refine_mapping_by_geometry(pairs, labels, ja_rest, js_rest)
        if pairs2 != pairs:
            print("\nMapping after auto-mapping:")
            for si, ai, sn, an in pairs2:
                print(f"  {sn:<16} -> {an:<20} ({ai})")
        compute_joint_offsets(pairs2, labels, ja_rest, Ra_rest, js_rest, offset_names)
        return
    if args.auto_mapping:
        pairs = refine_mapping_by_geometry(pairs, labels, ja_rest, js_rest)

    if len(pairs) < 8:
        print("\nABORT: too few joints mapped to fit anything meaningful.")
        print("Run --list_bones, then fix SMPL_TO_ANNY at the top of this file.")
        sys.exit(1)

    smpl_idx = [p[0] for p in pairs]
    anny_idx = [p[1] for p in pairs]
    bone_of = {p[0]: p[3] for p in pairs}
    offsets = compute_joint_offsets(pairs, labels, ja_rest, Ra_rest, js_rest, offset_names)

    if not args.pkl:
        print("\nNo --pkl given; mapping check only. Supply a 3DPW .pkl to fit.")
        return

    if not os.path.isfile(args.pkl):
        print(f"ERROR: not found: {args.pkl}"); sys.exit(1)
    with open(args.pkl, 'rb') as f:
        seq = pickle.load(f, encoding='latin1')
    print(f"\nloaded {args.pkl}")
    print(f"  keys: {sorted(seq.keys())}")

    person = args.person
    n_people = len(seq['poses'])
    if person >= n_people:
        print(f"ERROR: --person {person} but the sequence has {n_people} subject(s)"); sys.exit(1)
    poses = np.asarray(seq['poses'][person])          # [T, 72]
    betas = np.asarray(seq['betas'][person])[:10]     # [10]
    T = poses.shape[0]
    gender = read_gender(seq, person)
    print(f"  {n_people} subject(s); using person {person}, gender {gender!r}; {T} frames total")

    frames = find_valid_frames(seq, person, T, args.max_frames, args.frame_stride)
    if not frames:
        print("\nABORT: no valid frames found in this sequence (all NaN / "
              "campose_valid=False). Try a different .pkl.")
        sys.exit(1)
    print(f"  fitting frames: {frames}")

    # ---- targets -----------------------------------------------------------
    trans_all = np.asarray(seq['trans'][person])
    smpl_joints = None
    if smpl is not None:
        def smpl_joints(t):
            p = torch.tensor(poses[t], dtype=torch.float32, device=device).reshape(1, 72)
            b = torch.tensor(betas, dtype=torch.float32, device=device).reshape(1, 10)
            with torch.no_grad():
                so = smpl(global_orient=p[:, :3], body_pose=p[:, 3:],
                          betas=b, transl=torch.zeros(1, 3, device=device))
            return so.joints[0, :24].cpu().numpy()

    target_mode = args.target
    jp = None
    if target_mode == 'gt':
        if 'jointPositions' in seq:
            jp = np.asarray(seq['jointPositions'][person])
            if jp.ndim == 2 and jp.shape[1] == 72 and jp.shape[0] == T:
                jp = jp.reshape(T, 24, 3)
            elif not (jp.ndim == 3 and jp.shape[1:] == (24, 3)):
                print(f"  jointPositions has shape {jp.shape}, expected [T,72] - falling back to --target smpl")
                jp = None
        else:
            print("  no 'jointPositions' in this .pkl - falling back to --target smpl")
        if jp is None:
            target_mode = 'smpl'
    if target_mode == 'smpl' and smpl_joints is None:
        print("ERROR: --target smpl needs --smpl_model_path"); sys.exit(1)

    if target_mode == 'gt':
        def target_joints(t):
            # 3DPW jointPositions are world-space (= SMPL joints + trans);
            # remove trans so the fit lives in the transl=0 model frame that
            # render_anny_fit.py re-adds.
            return jp[t] - trans_all[t]
        print("  target: 3DPW jointPositions (gendered ground truth) minus trans")
        if smpl_joints is not None:
            t0 = frames[0]
            diff = np.linalg.norm(smpl_joints(t0) - target_joints(t0), axis=-1) * 1000
            print(f"  neutral-SMPL vs GT joints on frame {t0}: mean {diff.mean():.1f} mm, "
                  f"max {diff.max():.1f} mm ({SMPL_JOINT_NAMES[int(diff.argmax())]})")
            if diff.mean() > 200:
                print("  WARNING: that is far more than a gender-model difference; the "
                      "jointPositions convention may not be 'SMPL joints + trans'. "
                      "Check, or rerun with --target smpl.")
    else:
        target_joints = smpl_joints
        print("  target: neutral SMPL model forward with the sequence betas (v1 behaviour)")

    # ---- shape setup --------------------------------------------------------
    shape_init = np.full(len(pheno_labels), 0.5, dtype=np.float32)
    if args.free_phenotypes.strip().lower() == 'all':
        free_names = set(pheno_labels)
    else:
        free_names = {s.strip() for s in args.free_phenotypes.split(',') if s.strip()}
    unknown = free_names - set(pheno_labels)
    if unknown:
        print(f"ERROR: unknown phenotype(s) in --free_phenotypes: {sorted(unknown)}; "
              f"choose from {pheno_labels}"); sys.exit(1)
    gender_fixed = False
    if 'gender' in pheno_labels:
        if args.fix_gender and gender in ('m', 'f'):
            shape_init[pheno_labels.index('gender')] = 1.0 if gender == 'm' else 0.0
            free_names.discard('gender')
            gender_fixed = True
            print(f"  gender phenotype fixed at {shape_init[pheno_labels.index('gender')]:.1f} "
                  f"from 3DPW gender {gender!r} (MakeHuman convention 0=female, 1=male - "
                  f"if the rendered body is the wrong sex, that convention is wrong: "
                  f"use --fix_gender 0)")
        else:
            free_names.add('gender')
    free_mask = np.array([n in free_names for n in pheno_labels], dtype=bool)
    print(f"  free phenotypes: {[n for n in pheno_labels if n in free_names]}; "
          f"fixed: {[n for n in pheno_labels if n not in free_names]}")
    print(f"  pose regulariser {args.pose_reg}, shape regulariser {args.shape_reg}, "
          f"free bones: {args.free_bones}, virtual joints: {sorted(SMPL_JOINT_NAMES[i] for i in offsets)}")
    fit_kw = dict(lr=args.lr, pose_reg=args.pose_reg, shape_reg=args.shape_reg,
                  offsets=offsets, free_bones=args.free_bones)

    # ---- PASS 1: every frame, shape free, warm-started from the previous frame
    print(f"\n=== PASS 1: fitting {len(frames)} frames (shape free, "
          f"{'warm-started' if args.warm_start else 'cold'}) ===")
    results = []
    prev = None
    for k, t in enumerate(frames):
        j_t = target_joints(t)
        if np.isnan(j_t).any():
            print(f"[frame {t}] target joints contain NaN despite the valid-frame filter - skipping.")
            continue
        it = tuple(args.warm_iters) if prev is not None else tuple(args.iters)
        r = fit_frame(anny_model, j_t, anny_idx, smpl_idx, device, iters=it,
                      verbose=(k == 0), shape_init=shape_init, free_shape=free_mask,
                      init=prev, **fit_kw)
        # Never let a NaN fit into the output file: a saved .npz full of NaN
        # looks like valid training data until it silently poisons a run.
        if not np.isfinite(r['joint_err_mm']) or not np.isfinite(r['anny_rotmat']).all():
            print(f"[frame {t}] produced NaN/Inf - DISCARDED, next frame starts cold.")
            prev = None
            continue
        r['frame'] = t
        results.append(r)
        if args.warm_start:
            prev = {'rot6d': r['_rot6d'], 'transl': r['transl'], 'shape_logit': r['_shape_logit']}
        print(f"[frame {t}] ({k+1}/{len(frames)}) err {r['joint_err_mm']:.1f} mm  "
              + " ".join(f"{k2[:4]}={v2:.2f}" for k2, v2 in r['anny_shape_named'].items()
                         if free_mask[pheno_labels.index(k2)]))

    if not results:
        print("\nABORT: every candidate frame failed. Try a different sequence "
              "or check the SMPL model file.")
        sys.exit(1)

    # ---- PASS 2: one body per person (median), pose-only refit from the pass-1 solution
    shared = None
    if args.shared_shape and len(results) > 1:
        sub = results
        if args.shape_frames > 0 and args.shape_frames < len(results):
            step = max(1, len(results) // args.shape_frames)
            sub = results[::step][:args.shape_frames]
        shared = np.median(np.stack([r['anny_shape'] for r in sub]), axis=0)
        print(f"\n=== PASS 2: refitting {len(results)} frames with the shared body "
              f"(median of {len(sub)} frames), {args.refine_iters} pose iterations each ===")
        print("  shared body: " + "  ".join(f"{k2}={v2:.2f}" for k2, v2 in zip(pheno_labels, shared)))
        for i, r in enumerate(results):
            r2 = fit_frame(anny_model, r['target_joints'], anny_idx, smpl_idx, device,
                           iters=(0, args.refine_iters, 0), verbose=False,
                           fixed_shape=shared, init={'rot6d': r['_rot6d'], 'transl': r['transl']},
                           **fit_kw)
            if not np.isfinite(r2['joint_err_mm']) or not np.isfinite(r2['anny_rotmat']).all():
                print(f"  [frame {r['frame']}] pass 2 produced NaN - keeping the pass-1 result "
                      f"(its own shape, not the shared one)")
                continue
            print(f"  [frame {r['frame']}] {r['joint_err_mm']:.1f} -> {r2['joint_err_mm']:.1f} mm")
            r2['frame'] = r['frame']
            results[i] = r2

    if not results:
        print("\nABORT: every candidate frame failed. Try a different sequence "
              "or check the SMPL model file.")
        sys.exit(1)

    errs = [r['joint_err_mm'] for r in results]
    print(f"\nfitted {len(results)} frames; joint error "
          f"mean {np.mean(errs):.1f} mm, max {np.max(errs):.1f} mm")
    print("INTERPRETATION: <30mm is a good fit; >80mm means the mapping or the")
    print("optimisation is wrong - do NOT train on those frames.")

    # Per-joint table: the diagnostic for wrong correspondences.
    pj = np.stack([r['per_joint_err_mm'] for r in results])          # [N, 24]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)              # all-NaN columns = unmapped joints
        mean_pj = np.nanmean(pj, axis=0)
    order = [i for i in np.argsort(-np.nan_to_num(mean_pj, nan=-1.0)) if np.isfinite(mean_pj[i])]
    print("\nPER-JOINT mean error over the sequence (worst first). A flat table = good")
    print("correspondences; one or two joints far above the rest = those joints are")
    print("mapped to the wrong Anny bone (fix SMPL_TO_ANNY / see --check_mapping).")
    for i in order:
        print(f"  {SMPL_JOINT_NAMES[i]:<16} -> {bone_of.get(i, '?'):<16} {mean_pj[i]:6.1f} mm")

    fr = np.array([r['frame'] for r in results])
    # Everything render_anny_fit.py needs to put the mesh back on the photo:
    #   world = anny_verts + trans_smpl[t]   (we fit with SMPL transl=0)
    #   cam   = cam_poses[t] @ world ;  pixels = cam_intrinsics @ cam
    # 3DPW's 30 Hz arrays (poses, trans, cam_poses) are indexed by IMAGE
    # number, so the image for pose frame t is image_{t:05d}.jpg. The pickle's
    # 'img_frame_ids' is the 60 Hz index map (0,2,4,...) - stored separately
    # for reference, never used as an image number.
    cam_poses = np.asarray(seq['cam_poses']) if 'cam_poses' in seq else None
    cam_K = np.asarray(seq['cam_intrinsics']) if 'cam_intrinsics' in seq else None
    ids60 = np.asarray(seq['img_frame_ids']) if 'img_frame_ids' in seq else None
    np.savez(args.out,
             anny_rotmat=np.stack([r['anny_rotmat'] for r in results]),
             anny_shape=np.stack([r['anny_shape'] for r in results]),
             transl=np.stack([r['transl'] for r in results]),
             joint_err_mm=np.array(errs, dtype=np.float32),
             per_joint_err_mm=pj.astype(np.float32),
             target_joints=np.stack([r['target_joints'] for r in results]),
             anny_joints=np.stack([r['anny_joints'] for r in results]),
             frames=fr,
             img_frame_ids=fr,                                    # IMAGE numbers (== frames)
             pose60hz_ids=ids60[fr] if ids60 is not None and len(ids60) > fr.max() else fr,
             trans_smpl=trans_all[fr],
             cam_poses=cam_poses[fr] if cam_poses is not None else np.zeros((len(fr), 4, 4)),
             cam_intrinsics=cam_K if cam_K is not None else np.eye(3),
             seq_name=os.path.splitext(os.path.basename(args.pkl))[0],
             person=person,
             gender=gender if gender is not None else '',
             gender_fixed=gender_fixed,
             target_mode=target_mode,
             shared_shape=shared if shared is not None else np.zeros(len(pheno_labels)),
             phenotype_labels=np.array(pheno_labels),
             free_phenotypes=free_mask,
             smpl_idx=np.array(smpl_idx), anny_idx=np.array(anny_idx),
             offset_joints=np.array(sorted(SMPL_JOINT_NAMES[i] for i in offsets)),
             joint_offsets_local=np.stack([offsets.get(i, np.zeros(3, np.float32)) for i in smpl_idx]),
             free_bones=args.free_bones, version='2.1',
             anny_bone_names=np.array([bone_of[i] for i in smpl_idx]))
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
