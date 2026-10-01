# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

from argparse import ArgumentParser
import torch
from datasets.bedlam import BEDLAM
from datasets.ehf import EHF
from datasets.threedpw import THREEDPW
from model import Model
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import time
import numpy as np
import smplx
from utils import perspective_projection, log_depth, focal_length_normalization, render_meshes, denormalize_rgb, SMPLX_DIR, AverageMeter, compute_prf1, match_2d_greedy, SMPLX2SMPL_REGRESSOR
from smplx.joint_names import JOINT_NAMES
import roma
from torch.utils.tensorboard import SummaryWriter
from loss import Loss
from PIL import Image
import pickle
import torch
import anny
# --- CUSTOM ANNYONE DATALOADER HACK (V3: NATIVE) ---
from datasets.annyone import AnnyOne
from datasets.bedlam import collate_fn as raw_collate
import csv
import json
import datetime

# The --val_data loaders (BEDLAM/EHF/THREEDPW, built in main()) and the BEDLAM
# training branch refer to `collate_fn`, which was never defined in this file
# (only the `raw_collate` alias above), so they raised NameError. It is the
# same function from datasets.bedlam.
collate_fn = raw_collate


# ---------------------------------------------------------------------------
# AnnyOne tail splits
# ---------------------------------------------------------------------------
# Layout of the AnnyOne index range [0, n_total):
#
#     [0, val_start)            train
#     [val_start, test_start)   validation  (--val_anny_n samples)
#     [test_start, n_total)     test        (--test_anny_n samples)
#
# With --test_anny_n 0 there is no test split (test_start == n_total) and the
# validation set is the last --val_anny_n samples, exactly as before.
def anny_split_ranges(n_total, val_n, test_n):
    test_start = max(0, n_total - test_n) if test_n > 0 else n_total
    val_start = max(0, test_start - val_n) if val_n > 0 else test_start
    return val_start, test_start


def anny_trained_end(n_total, a):
    """Exclusive end of the AnnyOne index range a checkpoint was TRAINED on
    (training always uses a prefix [0, end)), reconstructed from the args
    Namespace saved inside the checkpoint. None if it cannot be determined."""
    if a is None or getattr(a, 'train_data', None) != 'AnnyOne':
        return None
    train_n = int(getattr(a, 'train_n', 0) or 0)
    val_n = int(getattr(a, 'val_anny_n', 0) or 0)
    test_n = int(getattr(a, 'test_anny_n', 0) or 0)
    if getattr(a, 'eval_only', 0):
        return None  # an eval-only run never saves checkpoints; be safe
    if train_n > 0:
        return min(train_n, n_total)
    if test_n > 0:
        return anny_split_ranges(n_total, val_n, test_n)[0]
    if val_n > 0:
        return n_total - val_n
    return n_total


# Column order of results/<run_name>.csv (one row per evaluate() call).
RESULT_FIELDS = [
    'timestamp', 'run_name', 'checkpoint', 'split', 'dataset', 'n_samples',
    'n_images_scored', 'epoch', 'iter',
    'PVE', 'PA-PVE', 'MPJPE', 'PA-MPJPE', 'mpjpe_joints',
    'precision', 'recall', 'F1',
    'n_gt_humans', 'n_matched', 'n_missed', 'n_false_pos',
    'n_seen_in_training',
]

class Trainer(object):
    def __init__(self, model, loss, optimizer, device, args, best_val=1e5, scheduler=None,
                 teacher=None, kd_loss=None):
        self.model = model
        self.loss = loss
        self.device = device
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Knowledge distillation: frozen teacher backbone + KL loss module.
        # Both None unless --distill_teacher_ckpt was given.
        self.teacher = teacher
        self.kd_loss = kd_loss
        self.best_val = best_val
        self.current_epoch = 0
        self.current_iter = 0

        # Parametric 3D human models
        self.smplx_neutral_11 = smplx.create(SMPLX_DIR, 'smplx', gender='neutral', use_pca=False, flat_hand_mean=True, num_betas=11).to(self.device)
        self.smpl_male_10 = smplx.create(SMPLX_DIR, 'smpl', gender='male').to(self.device)
        self.smpl_female_10 = smplx.create(SMPLX_DIR, 'smpl', gender='female').to(self.device)
        with open(SMPLX2SMPL_REGRESSOR, 'rb') as f:
            self.smplx2smpl_regressor = torch.from_numpy(pickle.load(f)['matrix'].astype(np.float32)).to(self.device)
        
        # Anny body model for native GT mesh generation        
        self.anny_body_model = anny.create_fullbody_model(
            remove_unattached_vertices=False,
            all_phenotypes=True,
        ).float().to(self.device)
        self.anny_body_model.set_skinning_method('lbs')

        self.args.log_dir = os.path.join(self.args.save_dir, self.args.name)
        os.makedirs(self.args.log_dir, exist_ok=True)

        self.args.ckpt_dir = os.path.join(self.args.log_dir, 'checkpoints')
        os.makedirs(self.args.ckpt_dir, exist_ok=True)

        self.args.visu_dir = os.path.join(self.args.log_dir, 'visu')
        os.makedirs(self.args.visu_dir, exist_ok=True)

        self.writer = SummaryWriter(self.args.log_dir)

    
    def prepare_gt(self, y):
        target = {}

        bs, nhmax = y['valid_humans'].shape

        # Valid humans
        valid_h = y['valid_humans']  # [bs,nh_max]
        idx_h = torch.where(valid_h)  # tuple of length=2
        nhv = int(valid_h.sum())
        K = y['K'][idx_h[0]]

        has_smplx_params = 0
        if 'smplx_vertices' in y:
            # EHF - only one person
            verts = y['smplx_vertices'].reshape(1, -1, 3)
            jts = self.smplx_neutral_11.J_regressor @ verts
        elif 'smpl_root_pose' in y:
            # 3DPW - eval only
            out = self.smpl_male_10(
                global_orient=y['smpl_root_pose'][idx_h[0], idx_h[1]].reshape(-1, 3),
                body_pose=y['smpl_body_pose'][idx_h[0], idx_h[1]].reshape(-1, 23*3),
                betas=y['smpl_shape'][idx_h[0], idx_h[1]].reshape(-1, 10),
                transl=y['smpl_transl'][idx_h[0], idx_h[1]].reshape(-1, 3),
                )
            verts, jts = out.vertices.reshape(nhv, -1, 3), out.joints.reshape(nhv, -1, 3)

            # update verts/joints if this is not the right gender
            if int(y['smpl_gender_id'].max()) == 2:
                out_female = self.smpl_female_10(
                    global_orient=y['smpl_root_pose'][idx_h[0], idx_h[1]].reshape(-1, 3),
                    body_pose=y['smpl_body_pose'][idx_h[0], idx_h[1]].reshape(-1, 23*3),
                    betas=y['smpl_shape'][idx_h[0], idx_h[1]].reshape(-1, 10),
                    transl=y['smpl_transl'][idx_h[0], idx_h[1]].reshape(-1, 3),
                )
                idx = torch.where(y['smpl_gender_id'] == 2)[1]
                verts[idx] = out_female.vertices.reshape(nhv, -1, 3)[idx]
                jts[idx] = out_female.joints.reshape(nhv, -1, 3)[idx]
        elif 'smplx_root_pose' in y:
            # SMPLX forward on valid humans only - BEDLAM
            has_smplx_params = 1                                     # ✅ FIXED: was wrongly indented inside the elif
            out = self.smplx_neutral_11(
                global_orient=y['smplx_root_pose'][idx_h[0], idx_h[1]].reshape(-1, 3),
                body_pose=y['smplx_body_pose'][idx_h[0], idx_h[1]].reshape(-1, 21*3),
                jaw_pose=y['smplx_jaw_pose'][idx_h[0], idx_h[1]].reshape(-1, 3),
                leye_pose=y['smplx_leye_pose'][idx_h[0], idx_h[1]].reshape(-1, 3),
                reye_pose=y['smplx_reye_pose'][idx_h[0], idx_h[1]].reshape(-1, 3),
                left_hand_pose=y['smplx_left_hand_pose'][idx_h[0], idx_h[1]].reshape(-1, 15*3),
                right_hand_pose=y['smplx_right_hand_pose'][idx_h[0], idx_h[1]].reshape(-1, 15*3),
                betas=y['smplx_shape'][idx_h[0], idx_h[1]].reshape(-1, 11),
                transl=y['smplx_transl'][idx_h[0], idx_h[1]].reshape(-1, 3),
                expression=self.smplx_neutral_11.expression.repeat(nhv, 1),
                )
            verts, jts = out.vertices.reshape(nhv, -1, 3), out.joints.reshape(nhv, -1, 3)  # ✅ FIXED: was outside the elif
        
        elif 'anny_pose' in y:
            raw_anny_pose = y['anny_pose'][idx_h[0], idx_h[1]].to(self.device)  # [nhv, 164, 4, 4]
            pose_163 = raw_anny_pose[:, :163]  # [nhv, 163, 4, 4] — keep as 4x4!

            anny_rotmats = pose_163[:, :, :3, :3]  # [nhv, 163, 3, 3]

            # ✅ FIX (shape supervision): use the dataset's anny_shape as GT
            # phenotypes instead of hardcoded zeros, so GT meshes match the
            # rendered bodies and the shape head gets a real target.
            # CONFIRMED via check_shape.py: anny_shape is [bs, nh, 11] in [0,1],
            # ordered exactly like self.anny_body_model.phenotype_labels:
            #   0 gender  1 age  2 muscle  3 weight  4 height  5 proportions
            #   6 cupsize 7 firmness  8 african  9 asian  10 caucasian
            # All 11 are real phenotypes the body model accepts, so forward
            # ALL of them (the old code dropped 6-10, corrupting the GT mesh).
            pheno_labels = list(self.anny_body_model.phenotype_labels)
            n_pheno = len(pheno_labels)                       # 11
            shape_vec = torch.zeros((nhv, n_pheno), device=self.device)
            use_shape = bool(getattr(self.args, 'use_anny_shape', 0)) and ('anny_shape' in y)
            if use_shape:
                raw_shape = y['anny_shape'][idx_h[0], idx_h[1]].to(self.device).float()
                raw_shape = raw_shape.reshape(nhv, -1)
                if not hasattr(self, '_printed_shape_dbg'):
                    self._printed_shape_dbg = True
                    print(f"[anny_shape] per-human shape={tuple(raw_shape.shape)}, "
                          f"min={raw_shape.min():.3f}, max={raw_shape.max():.3f}, "
                          f"phenotype_labels={pheno_labels}", flush=True)
                ns = min(raw_shape.shape[-1], n_pheno)
                shape_vec[:, :ns] = torch.clamp(raw_shape[:, :ns], 0.0, 1.0)

            # Forward every phenotype into the body model (positional order
            # matches phenotype_labels, confirmed above).
            _shape = {k: shape_vec[:, l] for l, k in enumerate(pheno_labels)}

            with torch.no_grad():
                anny_out = self.anny_body_model(
                    pose_parameters=pose_163,        
                    phenotype_kwargs=_shape,
                )
            
            verts = anny_out['vertices']   # [nhv, V, 3]
            
            # ✅ FIX: Extract the actual posed 3D joint translations from the homogeneous matrices
            jts = anny_out['bone_poses'][:, :, :3, -1]  # [nhv, 163, 3]
            
            # GT supervision signals use the 3x3 rotmats
            anny_rotmats_flat = anny_rotmats.reshape(-1, 3, 3)
            U, _, Vh = torch.linalg.svd(anny_rotmats_flat)
            
            # ✅ FIX: Enforce valid SO(3) rotation matrices to prevent arccos NaNs!
            det = torch.linalg.det(U @ Vh)
            D = torch.ones_like(U[:, :, 0])
            D[:, 2] = det
            anny_rotmats_ortho = U @ torch.diag_embed(D) @ Vh

            gt_rotmats_final = anny_rotmats_ortho.reshape(nhv, 163, 3, 3)
            # --- HELPER-JOINT MASKING (training side) ---
            # When --mask_helper_joints_train is on, the model forces the 88
            # helper bones to identity in its forward pass; mask the GT to
            # identity for the same joints so pred==GT there and the rotmat
            # loss contributes ZERO on them, instead of an irreducible floor.
            # GT verts/jts above stay built from the FULL unmasked pose — the
            # geometry supervision keeps its real targets.
            if getattr(self.args, 'mask_helper_joints_train', 0) and hasattr(self.model, 'useful_rotmat'):
                m = self.model.useful_rotmat.reshape(1, -1, 1, 1).to(
                    gt_rotmats_final.dtype).to(gt_rotmats_final.device)
                eye = torch.eye(3, device=gt_rotmats_final.device,
                                dtype=gt_rotmats_final.dtype).reshape(1, 1, 3, 3)
                gt_rotmats_final = m * gt_rotmats_final + (1.0 - m) * eye

            target['rotmat'] = gt_rotmats_final
            target['rotvec'] = roma.rotmat_to_rotvec(
                gt_rotmats_final.reshape(-1, 3, 3)
            ).reshape(nhv, 163, 3)
            target['shape'] = shape_vec  # ✅ real phenotype GT (zeros if --use_anny_shape 0)
            has_smplx_params = 0
        else:
            return None

        j2d = perspective_projection(jts, K)
        v2d = perspective_projection(verts, K)

        # Translation of the primary keypoint
        if 'anny_pose' in y:
            root_joint_idx = 0
        else:
            root_joint_idx = JOINT_NAMES.index(self.args.person_center)
            
        target['transl'] = jts[:, root_joint_idx]
        target['transl_pelvis'] = jts[:, 0]
        # ✅ FIX: Clamp distance to prevent log(0) NaNs
        target['dist'] = torch.clamp(jts[:, 0, -1], min=0.1)

        # We may predict dist in log space, or normalized values.
        if self.model.nearness:
            non_euclidean_dist = log_depth(target['dist'])
            # Normalise by focal
            focal = K[:, 0, 0]
            non_euclidean_dist = focal_length_normalization(non_euclidean_dist, focal, fovn=60, img_size=self.model.img_size)
            target['dist_postprocessed'] = non_euclidean_dist

        # Fill in target
        target['v3d'] = verts
        target['j3d'] = jts
        target['j2d'] = j2d
        target['v2d'] = v2d

        # Creating the target heatmap for the primary keypoint
        n_patch = args.img_size // self.model.patch_size
        pk = target['transl'].unsqueeze(1)  # (nhv,3)
        pk_loc = perspective_projection(pk, K).squeeze(1)
        pk_coarse_loc = (pk_loc // self.model.patch_size).int()  # (nhv,2)
        pk_idx = torch.clamp(pk_coarse_loc, 0, n_patch - 1)     # (nhv,2)
        pk_offset = (pk_loc - (pk_idx + 0.5) * self.model.patch_size) / self.model.patch_size

        # ✅ FIX (offset spikes): only keep humans whose person-center is
        # in front of the camera AND projects inside the image. Otherwise
        # pk_loc can be huge/flipped and pk_offset explodes (values in the
        # thousands were polluting the offset loss).
        img_size = self.args.img_size
        in_front = target['transl'][:, 2] > 0.1                     # (nhv,)
        in_image = (pk_loc[:, 0] >= 0) & (pk_loc[:, 0] < img_size) & \
                   (pk_loc[:, 1] >= 0) & (pk_loc[:, 1] < img_size)  # (nhv,)
        is_valid = (in_front & in_image).float()

        # Belt-and-braces: a correct in-image offset is in [-0.5, 0.5];
        # clamp slightly wider so any residual edge case cannot explode.
        pk_offset = torch.clamp(pk_offset, min=-1.0, max=1.0)

        # Scores & updating valid_humans according to occlusion + validity
        scores = torch.zeros((bs, n_patch, n_patch)).to(self.device)
        visible_humans = is_valid.clone()                            # was: ones(nhv)
        for k in range(nhv):
            i = int(idx_h[0][k])
            j = int(idx_h[1][k])
            if visible_humans[k] == 0:
                # out-of-frame / behind-camera human: drop it and do NOT
                # write a positive into the detection heatmap
                valid_h[i, j] = 0
                continue
            _x = pk_idx[k, 1]
            _y = pk_idx[k, 0]
            if scores[i, _x, _y] == 1:
                valid_h[i, j] = 0
                visible_humans[k] = 0
            else:
                scores[i, _x, _y] = 1

        target['loc'] = pk_loc
        target['offset'] = pk_offset

        # Only set rotvec/rotmat/shape here for SMPL-X batches.
        # For Anny batches these are already set in the elif block above.
        if has_smplx_params:
            target['rotvec'] = torch.cat([y['smplx_root_pose'],
                                        y['smplx_body_pose'],
                                        y['smplx_left_hand_pose'],
                                        y['smplx_right_hand_pose'],
                                        y['smplx_jaw_pose']], 2)[idx_h[0], idx_h[1]]
            target['rotmat'] = roma.rotvec_to_rotmat(target['rotvec'])
            target['shape'] = y['smplx_shape'][idx_h[0], idx_h[1]]

        # Update with visibility indices
        _target = {}
        idx_vis = torch.where(visible_humans)[0]
        # ✅ FIX: nothing left to supervise -> skip this batch
        # (train_n_iters already handles gt is None)
        if idx_vis.numel() == 0:
            return None
        _target['idx'] = tuple([
            idx_h[0].to(self.device)[idx_vis],
            pk_idx[:, 1].to(self.device)[idx_vis],
            pk_idx[:, 0].to(self.device)[idx_vis],
            torch.zeros_like(idx_h[0].to(self.device)[idx_vis])
        ])
        _target['scores'] = scores
        _target['K'] = y['K']
        for k, v in target.items():
            _target[k] = v[idx_vis]  # discard invisible humans due to occlusion

        return _target

    def fit(self, data_train, l_data_val):

        start_epoch = 0
        for epoch in range(start_epoch, self.args.max_epochs):
            
            # Training
            timer_end = time.time()
            self.train_n_iters(data_train)
            train_n_iters_time = time.time() - timer_end

            # Decay LR once per epoch. A constant LR that was right for training
            # fresh heads from random init becomes too aggressive once resumed
            # on a partially-trained model without decay: this caused repeated
            # NaN/Inf gradients and loss regression during the last resume
            # (see train_s1_resume / train_s2_resume logs). Cold-start runs still
            # get the full --learning_rate for --lr_decay_every epochs before the
            # first decay, so this is a no-op risk for fresh runs.
            if self.scheduler is not None:
                self.scheduler.step()
                print(f"[lr] epoch {self.current_epoch}: lr = "
                      f"{self.optimizer.param_groups[0]['lr']:.2e}", flush=True)

            # Checkpointing
            model_state_dict = self.model.state_dict()
            l_x = []
            for k in model_state_dict.keys(): # discard smpl_layer
                    if 'smpl_layer_' in k:
                        l_x.append(k)
            for x in l_x:
                model_state_dict.pop(x)

            save_dict = {'epoch': self.current_epoch,
                        'iter': self.current_iter,
                        'model_state_dict': model_state_dict,
                        'args': self.args}
            torch.save(save_dict, os.path.join(self.args.ckpt_dir, f"{self.current_epoch:05d}.pt"))

            # Cleaning old ckpt
            epochs = []
            for x in os.listdir(self.args.ckpt_dir):
                if '.pt' in x:
                    epoch = int(x.split('.pt')[0])
                    epochs.append(epoch)
            epochs.sort()
            epochs_to_keep = epochs[-self.args.nb_max_ckpt:]
            for x in epochs:
                fn = os.path.join(self.args.ckpt_dir, f"{x:05d}.pt")
                if x not in epochs_to_keep:
                    try:
                        os.remove(fn)
                    except:
                        print('trying to remove')

            # Evaluating (re-enabled): runs every --eval_freq epochs on the
            # held-out AnnyOne slice built in main(). Prints/logs pve, pa_pve,
            # precision, recall, f1_score, mpjpe, pa_mpjpe.
            timer_end = time.time()
            if len(l_data_val) > 0 and ((self.current_epoch + 1) % max(1, getattr(self.args, 'eval_freq', 1)) == 0):
                for data_val in l_data_val:
                    self.evaluate(data_val)
            evaluate_time = time.time() - timer_end

            # Flush metrcs to tensorboard
            self.writer.add_scalar(f"workload/train_n_iters", train_n_iters_time, self.current_epoch)
            self.writer.add_scalar(f"workload/evaluate", evaluate_time, self.current_epoch)
            self.writer.add_scalar(f"workload/ratio_trainVal", evaluate_time/(train_n_iters_time+evaluate_time), self.current_epoch)

            self.current_epoch += 1

        return 1
    
    def train_n_iters(self, data):
        # --- head warm-up freeze (see --freeze_heads_epochs) -----------------
        _fhe = int(getattr(self.args, 'freeze_heads_epochs', 0))
        _hk = getattr(self.model, '_teacher_head_keys', None)
        if _fhe > 0 and _hk:
            _frozen = self.current_epoch < _fhe
            _n = 0
            for _name, _p in self.model.named_parameters():
                if _name in _hk:
                    _p.requires_grad = not _frozen
                    _n += 1
            if self.current_epoch in (0, _fhe):
                print(f"[distill] epoch {self.current_epoch}: {_n} teacher-initialised head "
                      f"tensors {'FROZEN (warm-up)' if _frozen else 'UNFROZEN'}", flush=True)
        print(f"\nTRAIN: ")
        self.model.train()
        # model.train() recursively re-enables train mode everywhere, including
        # a frozen backbone, which would restart its BatchNorm/dropout updates.
        # requires_grad=False already blocks weight updates; this covers the rest.
        if getattr(self.args, 'freeze_backbone', 0):
            self.model.backbone.eval()
            # With partial unfreeze, the trainable tail blocks (+ final norm)
            # should be in train mode, not eval — put them back after the
            # backbone-wide eval() above.
            k = getattr(self.args, 'unfreeze_last_n_blocks', 0)
            if k > 0:
                blocks = self.model.backbone.encoder.blocks
                for blk in blocks[len(blocks) - min(k, len(blocks)):]:
                    blk.train()
                self.model.backbone.encoder.norm.train()

        meters = {k: AverageMeter(k) for k in ['workload/data', 'workload/batch', 'workload/ratio_data']}

        timer_end = time.time()
        for i, batch in enumerate(tqdm(data)):
            # ✅ FIX: honor --n_iters_per_epoch. Without this, one "epoch" is
            # the entire dataloader (145k iters ≈ 29h on the full dataset),
            # which starves checkpointing/eval and makes --start_2d_epoch
            # unreachable. Shuffle=True means each capped epoch sees a fresh
            # random slice of the data.
            if self.args.n_iters_per_epoch > 0 and i >= self.args.n_iters_per_epoch:
                break
            if batch is None or batch[0] is None:
                print(f"Batch {i} skipped due to dataloader crash!", flush=True) # <-- ADD THIS
                timer_end = time.time()
                continue
            x, y = batch

            y = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in y.items()}
            data_time = time.time() - timer_end

            # --- ADD THIS DEBUG BLOCK ---
            if i == 0:
                print(f"\n--- DEBUG BATCH ---", flush=True)
                print(f"Keys found in 'y': {list(y.keys())}", flush=True)
            # ----------------------------

            gt = self.prepare_gt(y=y)
            if gt is None:                  # ← FIX Bug 1
                timer_end = time.time()
                continue

            x = x.to(self.device)

            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                pred = self.model(x, is_training=True, idx=gt['idx'], K=gt['K'])
                pred['transl_pelvis'] = pred['transl_pelvis'].reshape(-1, 3)
                pred['transl'] = pred['transl'].reshape(-1, 3)
                loss, dict_loss = self.loss(pred, gt, epoch=self.current_epoch, img_size=self.args.img_size)

                # --- KNOWLEDGE DISTILLATION ---
                # Compare the STUDENT's backbone patch tokens against the
                # frozen TEACHER's, at the tap added in model.py (immediately
                # after patch decoding, before the projection heads). The
                # teacher runs under no_grad: it is never updated, and this
                # also avoids building a second backward graph through 300M+
                # frozen parameters.
                if self.teacher is not None and self.kd_loss is not None:
                    with torch.no_grad():
                        t_out = self.teacher(x, is_training=True,
                                             idx=gt['idx'], K=gt['K'])
                    kd = self.kd_loss(pred['backbone_feat'],
                                      t_out['backbone_feat'])
                    kd = torch.nan_to_num(kd, nan=0.0, posinf=0.0, neginf=0.0)
                    loss = loss + self.args.lambda_kd * kd
                    dict_loss['kd'] = kd

                    # --- OUTPUT-level distillation --------------------------
                    # Student and teacher ran on the SAME gt query locations
                    # (idx=gt['idx']), so their per-person outputs line up 1:1.
                    # Weight the terms like the task loss so the scale matches.
                    lko = float(getattr(self.args, 'lambda_kd_out', 0.0))
                    if lko > 0 and pred['rotmat'].shape == t_out['rotmat'].shape:
                        a = self.args
                        kd_rot = (pred['rotmat'] - t_out['rotmat'].detach()).abs().sum([1, 2, 3]).mean()
                        kd_shp = (pred['shape'] - t_out['shape'].detach()).abs().sum(-1).mean()
                        kd_dst = (pred['dist_postprocessed'].reshape(-1)
                                  - t_out['dist_postprocessed'].detach().reshape(-1)).abs().mean()
                        kd_out = (a.alpha_rotmat * kd_rot + a.alpha_shape * kd_shp
                                  + a.alpha_dist * kd_dst)
                        kd_out = torch.nan_to_num(kd_out, nan=0.0, posinf=0.0, neginf=0.0)
                        loss = loss + lko * kd_out
                        dict_loss['kd_out'] = kd_out
                    dict_loss['total'] = loss

            # NaN guard — skip corrupted batches
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss at iter {i}, skipping batch", flush=True)
                self.optimizer.zero_grad()
                timer_end = time.time()
                continue

            loss.backward()

            # --- ADD THIS GRADIENT GUARD ---
            has_nan_grad = False
            for param in self.model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"WARNING: NaN/Inf GRADIENTS at iter {i}, skipping optimizer step", flush=True)
                self.optimizer.zero_grad()
                timer_end = time.time()
                continue
            # --------------------------------

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            batch_time = time.time() - timer_end

            meters['workload/data'].update(data_time)
            meters['workload/batch'].update(batch_time)
            meters['workload/ratio_data'].update(data_time / batch_time)
            
            for k, v in dict_loss.items():
                k_name = f"loss/{k}"
                if k_name not in meters:
                    meters[k_name] = AverageMeter(k_name)
                meters[k_name].update(dict_loss[k].item())

            if i % self.args.log_freq == 0 and 'loss/total' in meters:  # ← FIX Bug 2
                # Create a dynamic string of ALL tracked losses
                loss_str = " | ".join([f"{k.replace('loss/', '')}: {v.avg:.4f}" for k, v in meters.items() if 'loss/' in k])
                print(f"EPOCH={self.current_epoch:03d} - i={i:05d}/{len(data):05d} -> {loss_str}")
                
                for k, v in meters.items():
                    self.writer.add_scalar(f"{k}", v.avg, self.current_iter)
                self.writer.flush()
                sys.stdout.flush()

            self.current_iter += 1
            timer_end = time.time()

        return 1

    def _write_results(self, row):
        """Append one evaluation row to <results_dir>/<run_name>.csv (header
        created if the file is new) and to <results_dir>/all_results.jsonl.
        Never raises: a failed write must not kill a training run."""
        try:
            rdir = getattr(self.args, 'results_dir', 'results') or 'results'
            if not os.path.isabs(rdir):
                rdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), rdir)
            os.makedirs(rdir, exist_ok=True)
            row = {k: ('' if v is None else v) for k, v in row.items()}
            safe_name = str(row.get('run_name', 'run')).replace(os.sep, '_')
            csv_path = os.path.join(rdir, f"{safe_name}.csv")

            fieldnames = list(RESULT_FIELDS)
            is_new = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
            if not is_new:
                # reuse the existing header so columns never shift
                with open(csv_path, newline='') as f:
                    header = next(csv.reader(f), None)
                if header:
                    fieldnames = header
            with open(csv_path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, restval='',
                                   extrasaction='ignore')
                if is_new:
                    w.writeheader()
                w.writerow(row)

            with open(os.path.join(rdir, 'all_results.jsonl'), 'a') as f:
                f.write(json.dumps(row) + '\n')
            print(f"[results] appended '{row.get('split')}' row to {csv_path}", flush=True)
        except Exception as e:
            print(f"[results] WARNING: could not write results: {type(e).__name__}: {e}",
                  flush=True)

    @torch.no_grad()
    def evaluate(self, data, split=None, checkpoint=None, epoch=None):
        """Run the metrics over `data` and append a row to the results files.
        split/checkpoint/epoch only label that row; when None they are derived
        from the dataset attributes and the trainer state."""
        print(f"\nEVAL: ")
        self.model.eval()

        # AnnyOne (or Subset-wrapped) datasets don't carry these attrs
        ds_name = getattr(data.dataset, 'name', 'annyone')
        ds_split = getattr(data.dataset, 'split', 'holdout')
        ds_subsample = getattr(data.dataset, 'subsample', 1)

        meters = {k: AverageMeter(k) for k in ['pve', 'pa_pve', 'precision', 'recall', 'f1_score',
                                               'mpjpe', 'pa_mpjpe'
                                               ]}
        count, miss, fp = 0, 0, 0
        n_scored = 0  # images that reached the metrics (not skipped)

        for i, (x,y) in enumerate(tqdm(data)):
            if x is None or y is None:
                continue
            # move tensor to device
            y = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in y.items()}
            
            # preprare gt by computing mesh and 3d/2d joints
            gt = self.prepare_gt(y=y)
            if gt is None:
                continue
            n_scored += 1

            # forward
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                x = x.to(self.device) # [bs,3,w,h]
                pred = self.model(x, is_training=False, K=gt['K'], 
                                  det_thresh=self.args.det_thresh, nms_kernel_size=self.args.nms_kernel_size)

            # match pred to gt - based on 2d bbox
            kp2d_gts = gt['j2d'].cpu().numpy()
            if len(pred) == 0:
                # no detections at all: every gt human is a miss
                count += len(kp2d_gts)
                miss += len(kp2d_gts)
                continue
            kp2d_preds = np.asarray([hum['j2d'].cpu().numpy()[:kp2d_gts.shape[1]] for hum in pred])
            bestMatch, falsePositives, misses = match_2d_greedy(kp2d_preds, kp2d_gts, np.ones_like(kp2d_gts[...,0]).astype(np.bool_))

            # detection metrics
            count += len(kp2d_gts)
            miss += len(misses)
            fp += len(falsePositives)

            # 3d metrics
            if len(bestMatch) > 0:
                for (pid, gid) in bestMatch:                    
                    # gt mesh centerex around pelvis
                    v3d = gt['v3d'][gid]
                    pelvis = gt['transl_pelvis'][gid].reshape(1,3)
                    v3d_ctx = v3d - pelvis

                    # pred mesh centerex around pelvis
                    v3d_hat = pred[pid]['v3d']
                    pelvis_hat = pred[pid]['transl_pelvis'].reshape(1,3)
                    v3d_hat_ctx = v3d_hat - pelvis_hat

                    # moving to smpl mesh for eval because gt are in smpl format
                    if v3d_ctx.shape[0] == 6890:
                        # The SMPL-X->SMPL regressor only applies to SMPL-X
                        # predictions (10475 verts). This model predicts Anny
                        # meshes, which have no vertex correspondence with the
                        # SMPL GT of 3DPW, so PVE/PA-PVE/MPJPE cannot be
                        # computed here (the matmul used to crash). Skip the 3D
                        # metrics; detection P/R/F1 above are still counted.
                        if v3d_hat_ctx.shape[0] != self.smplx2smpl_regressor.shape[1]:
                            if not getattr(self, '_warned_smpl_mismatch', False):
                                self._warned_smpl_mismatch = True
                                print(f"[eval] WARNING: GT is SMPL (6890 verts) but the "
                                      f"prediction has {v3d_hat_ctx.shape[0]} verts (not "
                                      f"SMPL-X). No Anny->SMPL vertex mapping exists in "
                                      f"train.py, so PVE/PA-PVE/MPJPE/PA-MPJPE are NOT "
                                      f"computed for {ds_name}; only precision/recall/F1.",
                                      flush=True)
                            continue
                        v3d_hat_ctx = (self.smplx2smpl_regressor @ v3d_hat_ctx)

                    # Per-Vertex Error
                    pve = ((torch.sqrt(((v3d_ctx - v3d_hat_ctx) ** 2).sum(-1))) * 1000).mean()
                    meters['pve'].update(pve.item())

                    # Procrustes-Aligned PVE
                    (R,t,s) = roma.rigid_points_registration(v3d_hat_ctx, v3d_ctx, compute_scaling=True)
                    pa_v3d_hat_ctx = s * (R.reshape(1,3,3) @ v3d_hat_ctx.reshape(-1,3,1)).reshape(-1,3) + t
                    pa_pve = ((torch.sqrt(((v3d_ctx - pa_v3d_hat_ctx) ** 2).sum(-1))) * 1000).mean()
                    meters['pa_pve'].update(pa_pve.item())

                    # MPJPE for 3DPW only (h36m-regressed joints)
                    if ds_name == '3dpw':
                        if i == 0:
                            # Can be download from https://github.com/nkolot/SPIN/blob/master/fetch_data.sh#L6C58-L6C78
                            self.J_regressor_h36m = torch.Tensor(np.load('models/smpl/J_regressor_h36m.npy')).to(self.device)
                            # https://github.com/nkolot/SPIN/blob/2476c436013055be5cb3905e4e4ecfa86966fac3/constants.py#L93C1-L95C31
                            self.H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
                            self.H36M_TO_J14 = self.H36M_TO_J17[:14]

                        # H36m joints
                        h36m = self.J_regressor_h36m @ v3d_ctx
                        h36m_hat = self.J_regressor_h36m @ v3d_hat_ctx

                        # center around h36m-pelvis
                        h36m_ctx = h36m - h36m[[0]]
                        h36m_hat_ctx = h36m_hat - h36m_hat[[0]]

                        # 14 joints only
                        h36m_ctx = h36m_ctx[self.H36M_TO_J14]
                        h36m_hat_ctx = h36m_hat_ctx[self.H36M_TO_J14]

                        # 17 joints only
                        # h36m_ctx = h36m_ctx[self.H36M_TO_J17]
                        # h36m_hat_ctx = h36m_hat_ctx[self.H36M_TO_J17]

                        # MPJPE
                        mpjpe = ((torch.sqrt(((h36m_ctx - h36m_hat_ctx) ** 2).sum(-1))) * 1000).mean()
                        meters['mpjpe'].update(mpjpe.item())

                        # PA-MPJPE
                        (R,t,s) = roma.rigid_points_registration(h36m_hat_ctx, h36m_ctx, compute_scaling=True)
                        pa_h36m_hat_ctx = s * (R.reshape(1,3,3) @ h36m_hat_ctx.reshape(-1,3,1)).reshape(-1,3) + t
                        pa_mpjpe = ((torch.sqrt(((h36m_ctx - pa_h36m_hat_ctx) ** 2).sum(-1))) * 1000).mean()
                        meters['pa_mpjpe'].update(pa_mpjpe.item())
                    else:
                        # Generic MPJPE/PA-MPJPE from the model's own joints
                        # (163 Anny joints), pelvis-centered, in mm.
                        j3d = gt['j3d'][gid] - gt['transl_pelvis'][gid].reshape(1, 3)
                        j3d_hat = pred[pid]['j3d'] - pred[pid]['transl_pelvis'].reshape(1, 3)
                        if j3d.shape[0] == j3d_hat.shape[0]:
                            mpjpe = ((torch.sqrt(((j3d - j3d_hat) ** 2).sum(-1))) * 1000).mean()
                            meters['mpjpe'].update(mpjpe.item())

                            (R, t, s) = roma.rigid_points_registration(j3d_hat, j3d, compute_scaling=True)
                            pa_j3d_hat = s * (R.reshape(1, 3, 3) @ j3d_hat.reshape(-1, 3, 1)).reshape(-1, 3) + t
                            pa_mpjpe = ((torch.sqrt(((j3d - pa_j3d_hat) ** 2).sum(-1))) * 1000).mean()
                            meters['pa_mpjpe'].update(pa_mpjpe.item())
            
            # log
            if i % self.args.log_freq == 0:
                precision, recall, f1_score = compute_prf1(count, miss, fp)
                print(f"i={i} - Recall={recall:.1f} - PVE={meters['pve'].avg:.1f} - PA-PVE={meters['pa_pve'].avg:.1f} - MPJPE={meters['mpjpe'].avg:.1f} - PA-MPJPE={meters['pa_mpjpe'].avg:.1f}")
                sys.stdout.flush()

            # visu
            if self.args.visu_to_save > 0 and i < self.args.visu_to_save:
                # image
                img_array = denormalize_rgb(x[0].cpu().numpy())
                focal = gt['K'][0,[0,1],[0,1]].cpu().numpy()
                princpt = gt['K'][0,[0,1],[-1,-1]].cpu().numpy()

                # gt
                gt_verts, gt_faces = [], []
                for j in range(len(gt['v3d'])):
                    gt_verts.append(gt['v3d'][j].cpu().numpy().reshape(-1,3))
                    gt_faces.append(self.smplx_neutral_11.faces if gt['v3d'][j].shape[0] == 10475 else self.smpl_male_10.faces)
                gt_rend_array = render_meshes(img_array.copy(), 
                                                gt_verts, 
                                                gt_faces,
                                                {'focal': focal, 'princpt': princpt})
                
                # pred
                pred_verts, pred_faces = [], []
                for j in range(len(pred)):
                    pred_verts.append(pred[j]['v3d'].cpu().numpy().reshape(-1,3))
                    pred_faces.append(
                        self.anny_body_model.faces if pred[j]['v3d'].shape[0] != 10475
                        else self.smplx_neutral_11.faces
                    )
                pred_rend_array = render_meshes(img_array.copy(), 
                                                pred_verts, 
                                                pred_faces,
                                                {'focal': focal, 'princpt': princpt})

                img = np.concatenate([img_array, pred_rend_array, gt_rend_array], 1)
                # Image.fromarray(img).save('img.jpg');ipdb.set_trace() # debugging
                Image.fromarray(img).save(os.path.join(self.args.visu_dir, f"img_epoch{self.current_epoch:04d}_{ds_name}_{i:04d}.jpg"))

        # final metrics
        print(f"***EVAL METRICS - {ds_name}-{ds_split}-{ds_subsample}***")
        precision, recall, f1_score= compute_prf1(count, miss, fp)
        meters['precision'].update(precision)
        meters['recall'].update(recall)
        meters['f1_score'].update(f1_score)
        for k, v in meters.items():
            self.writer.add_scalar(f"{ds_name}-{ds_split}-{ds_subsample}/{k}", v.avg, self.current_iter)
            print(f"    - {k}: {v.avg:.1f}")
        self.writer.flush() # https://github.com/pytorch/pytorch/issues/24234
        sys.stdout.flush()

        # ---- results as data: results/<run_name>.csv + results/all_results.jsonl
        if split is None:
            split = getattr(data.dataset, 'results_split', None)
        if split is None:
            if ds_name == '3dpw':
                split = '3dpw'
            elif ds_split == 'holdout':
                split = 'val'
            else:
                split = f"{ds_name}-{ds_split}"
        if checkpoint is None:
            checkpoint = getattr(self, 'eval_checkpoint', None)
            if checkpoint is None and not self.args.eval_only:
                # fit() saves this epoch's checkpoint just before evaluating
                checkpoint = os.path.join(self.args.ckpt_dir, f"{self.current_epoch:05d}.pt")
        if epoch is None:
            epoch = getattr(self, 'eval_epoch', None) if self.args.eval_only else self.current_epoch
        try:
            n_samples = len(data.dataset)
        except Exception:
            n_samples = ''

        def _m(k):  # '' when the metric was never measured (e.g. no matches)
            return round(float(meters[k].avg), 4) if meters[k].count > 0 else ''

        det = count > 0  # P/R/F1 are meaningless without any GT humans
        row = {
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'run_name': self.args.name,
            'checkpoint': checkpoint,
            'split': split,
            'dataset': ds_name,
            'n_samples': n_samples,
            'n_images_scored': n_scored,
            'epoch': epoch,
            'iter': self.current_iter,
            'PVE': _m('pve'),
            'PA-PVE': _m('pa_pve'),
            'MPJPE': _m('mpjpe'),
            'PA-MPJPE': _m('pa_mpjpe'),
            'mpjpe_joints': ('h36m14' if ds_name == '3dpw' else 'model_joints') if meters['mpjpe'].count > 0 else '',
            'precision': _m('precision') if det else '',
            'recall': _m('recall') if det else '',
            'F1': _m('f1_score') if det else '',
            'n_gt_humans': count,
            'n_matched': meters['pve'].count,
            'n_missed': miss,
            'n_false_pos': fp,
            'n_seen_in_training': getattr(data.dataset, 'n_seen_in_training', ''),
        }
        self._write_results(row)
        return meters['pve'].avg
    
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(pretrained_backbone=1, **vars(args))
    model = model.to(device)

    # Epoch / training args stored in the loaded checkpoint (if any). Used by
    # --eval_only to label result rows and to check split/train overlap.
    pretrained_epoch, pretrained_args = None, None

    # Load from a pretrained model
    if args.pretrained is not None and os.path.isfile(args.pretrained):
        print(f"Loading weights from {args.pretrained}", flush=True)
        ckpt = torch.load(args.pretrained, map_location='cpu')
        if isinstance(ckpt, dict):
            pretrained_epoch = ckpt.get('epoch', None)
            pretrained_args = ckpt.get('args', None)

        # Checkpoints from the original repo may not use the 'model_state_dict'
        # key, so locate the state dict rather than assuming.
        if 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt  # raw state dict
        print(f"[pretrained] checkpoint contains {len(sd)} tensors", flush=True)

        # ------------------------------------------------------------------
        # KEY REMAPPING
        # ------------------------------------------------------------------
        # multiHMR_672_L_anny was trained with the Multi_HMR class
        # (multi_hmr_anny/multi_hmr.py), which nests the DINOv2 ViT as
        #     self.encoder (Encoder) -> self.backbone (dinov2)  ->  "encoder.backbone.*"
        # while this Model (model.py) nests it the other way round:
        #     self.backbone (Dinov2Backbone) -> self.encoder (dinov2) -> "backbone.encoder.*"
        # The inner names are identical (both come from the same torch.hub
        # dinov2 module), so swapping the first two path components transfers
        # the whole ViT (~343 tensors / ~303M params).
        #
        # Nothing else transfers: the checkpoint's encoder.mlp_det /
        # mlp_fov_unique regress detection+FOV inside the encoder (this Model
        # takes K as input and has a separate mlp_classif), and its
        # decoder/mlp_pose/mlp_shape/mlp_dist stack is a different head
        # architecture from x_attention_head. Those train from scratch, which
        # is exactly what the original run did with load_only_backbone=1.
        if args.pretrained_remap:
            remapped, n_ren = {}, 0
            for k, v in sd.items():
                if k.startswith('encoder.backbone.'):
                    remapped['backbone.encoder.' + k[len('encoder.backbone.'):]] = v
                    n_ren += 1
                else:
                    remapped[k] = v
            if n_ren:
                print(f"[pretrained] remapped {n_ren} keys "
                      f"'encoder.backbone.*' -> 'backbone.encoder.*'", flush=True)
            sd = remapped

        # Optionally keep ONLY the backbone (the original repo's
        # --load_only_backbone). Everything else is left at its fresh init.
        if args.load_only_backbone:
            before = len(sd)
            sd = {k: v for k, v in sd.items() if k.startswith('backbone.')}
            print(f"[pretrained] load_only_backbone=1: kept {len(sd)}/{before} tensors",
                  flush=True)

        # ------------------------------------------------------------------
        # VERIFIED LOAD
        # ------------------------------------------------------------------
        # strict=False silently drops every key that does not match, so count
        # what actually lands before trusting it.
        msd = model.state_dict()
        matched = [k for k, v in sd.items()
                   if k in msd and hasattr(v, 'shape') and msd[k].shape == v.shape]
        log = model.load_state_dict(sd, strict=False)

        n_bb_model = sum(1 for k in msd if k.startswith('backbone.'))
        n_bb_loaded = sum(1 for k in matched if k.startswith('backbone.'))
        bb_pct = 100.0 * n_bb_loaded / max(n_bb_model, 1)
        loaded_params = sum(msd[k].numel() for k in matched)
        total_params = sum(v.numel() for v in msd.values())

        print(f"[pretrained] loaded {len(matched)}/{len(msd)} model tensors "
              f"({100.0*loaded_params/max(total_params,1):.1f}% of params)", flush=True)
        print(f"[pretrained] backbone: {n_bb_loaded}/{n_bb_model} tensors "
              f"({bb_pct:.1f}%)", flush=True)
        if log.unexpected_keys[:5]:
            print(f"[pretrained] unexpected examples: {log.unexpected_keys[:5]}", flush=True)

        # The backbone is the whole point of this transfer, so fail loudly if it
        # did not land rather than silently training a random ViT for a day.
        if bb_pct < args.min_backbone_match:
            raise RuntimeError(
                f"Only {bb_pct:.1f}% of the backbone was initialised from "
                f"{args.pretrained} (threshold --min_backbone_match "
                f"{args.min_backbone_match}). Inspect with:\n"
                f"  python inspect_ckpt.py --ckpt {args.pretrained} --compare\n"
                f"  python inspect_prefix.py --ckpt {args.pretrained} --prefix encoder"
            )

    # ----------------------------------------------------------------------
    # OPTIONAL BACKBONE FREEZE  (task step 2)
    # ----------------------------------------------------------------------
    # Keeps the loaded weights but excludes them from optimisation, so the
    # backbone acts as a fixed feature extractor and only the heads train.
    # Must happen BEFORE the optimizer is built.
    if args.freeze_backbone:
        n_frozen = sum(p.numel() for p in model.backbone.parameters())
        for p in model.backbone.parameters():
            p.requires_grad = False
        model.backbone.eval()  # also stop BN/dropout stats from drifting
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[freeze_backbone] froze {n_frozen:,} params; "
              f"{n_train:,} trainable params remain", flush=True)

        # PARTIAL UNFREEZE: re-enable the LAST N transformer blocks + final
        # norm. Early ViT blocks encode generic features (edges, textures)
        # that transfer as-is; later blocks are task-specific and benefit
        # from adapting to AnnyOne. Gradients stop at the first trainable
        # block, so most of the frozen ViT still runs backward-free.
        if args.unfreeze_last_n_blocks > 0:
            blocks = model.backbone.encoder.blocks
            n_blocks = len(blocks)
            k = min(args.unfreeze_last_n_blocks, n_blocks)
            n_unfrozen = 0
            for blk in blocks[n_blocks - k:]:
                for p in blk.parameters():
                    p.requires_grad = True
                    n_unfrozen += p.numel()
            for p in model.backbone.encoder.norm.parameters():
                p.requires_grad = True
                n_unfrozen += p.numel()
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[freeze_backbone] partial unfreeze: last {k}/{n_blocks} blocks "
                  f"+ final norm trainable again ({n_unfrozen:,} params); "
                  f"total trainable now {n_train:,}", flush=True)

    # Training-side helper-joint masking (see model.py / --mask_helper_joints_train).
    if getattr(args, 'mask_helper_joints_train', 0) and hasattr(model, 'mask_helper_joints_train'):
        model.mask_helper_joints_train = True
        print("[mask_helper_joints_train] 88 helper bones forced to identity during "
              "training; GT rotation targets masked to match (see prepare_gt)", flush=True)

    l_val_data = []
    assert len(args.val_split) == len(args.val_data) == len(args.val_subsample)
    for i in range(len(args.val_data)):
        val_data = DataLoader(eval(args.val_data[i])(split=f"{args.val_split[i]}", 
                                                training=0, 
                                                img_size=args.img_size,
                                                subsample=args.val_subsample[i], # for fast evaluation on a subsampled part of the validation
                                                n=args.val_n[i], # for debugging purpose only
                                                ),
                            batch_size=1,
                            num_workers=0,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=collate_fn,
                            )
        l_val_data.append(val_data)
    
    # Only optimise params that require grad — all of them normally, heads-only
    # when --freeze_backbone 1. Handing frozen params to Adam wastes optimizer
    # state and can still nudge them through weight decay.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    # Decay LR every --lr_decay_every epochs by --lr_decay_gamma. Needed for
    # resumed runs in particular — see the note in Trainer.fit(). Defaults
    # (30 epochs, x0.5) are a mild decay that barely affects the first ~30
    # epochs of a cold-start run.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_gamma
    )

    loss = Loss(args)
    # Configure per-joint loss weights from the model's bone names (no-op when
    # both boost flags are 1.0). Must run after both model and loss exist.
    if hasattr(loss, 'configure_joint_weights') and hasattr(model, 'body_model'):
        if getattr(args, 'boost_neck_weight', 1.0) != 1.0 or getattr(args, 'boost_hand_weight', 1.0) != 1.0:
            loss.configure_joint_weights(list(model.body_model.bone_labels))

    # ---------------- KNOWLEDGE DISTILLATION SETUP ----------------
    # --distill_teacher_ckpt turns this run into student training: `model` above
    # is the STUDENT (use --backbone dinov2_vits14 or _vitb14), and the teacher
    # is loaded frozen from one of our own checkpoints (use the ViT-L run).
    teacher, kd_loss = None, None
    if getattr(args, 'distill_teacher_ckpt', ''):
        from distill import KLDivergenceLoss, build_teacher
        if not os.path.isfile(args.distill_teacher_ckpt):
            raise FileNotFoundError(
                f"--distill_teacher_ckpt not found: {args.distill_teacher_ckpt}")
        teacher, teacher_dim = build_teacher(args.distill_teacher_ckpt, device)

        # --- initialise the STUDENT's heads from the TEACHER ---------------
        # Same Model class -> same key names. Copy everything outside the
        # backbone whose shape matches. The few layers that touch embed_dim
        # (1024 teacher vs 384 student) cannot transfer and stay random.
        if getattr(args, 'init_heads_from_teacher', 0):
            sd_t = teacher.state_dict()
            sd_s = model.state_dict()
            copied, skipped_shape, skipped_bb = [], [], 0
            for k, v in sd_t.items():
                if k.startswith('backbone.'):
                    skipped_bb += 1
                    continue
                if k in sd_s and tuple(sd_s[k].shape) == tuple(v.shape):
                    sd_s[k] = v.clone()
                    copied.append(k)
                else:
                    skipped_shape.append(k)
            model.load_state_dict(sd_s)
            # remembered for --freeze_heads_epochs (see train_n_iters)
            model._teacher_head_keys = set(copied)
            n_head = sum(1 for k in sd_s if not k.startswith('backbone.'))
            print(f"[distill] heads initialised from teacher: {len(copied)}/{n_head} "
                  f"non-backbone tensors copied ({100.0*len(copied)/max(n_head,1):.1f}%); "
                  f"{skipped_bb} backbone tensors intentionally left as-is", flush=True)
            if skipped_shape:
                print(f"[distill] {len(skipped_shape)} tensors NOT transferable "
                      f"(shape differs, embed_dim-dependent) - random init kept:", flush=True)
                for k in skipped_shape[:20]:
                    print(f"           {k}  teacher{tuple(sd_t[k].shape)} vs "
                          f"student{tuple(sd_s[k].shape) if k in sd_s else 'MISSING'}", flush=True)
            if len(copied) / max(n_head, 1) < 0.5:
                raise RuntimeError("Fewer than half of the head tensors transferred - "
                                   "teacher/student architectures differ more than expected.")
        # Width of the tokens the heads (and the KD tap) see. With --head_dim this
        # is the projector output, equal to the teacher width, so the KD loss
        # needs no projector of its own.
        student_dim = model.embed_dim
        if getattr(model, 'feat_proj', None) is not None:
            print(f"[distill] projector-fed heads: backbone {model.backbone_dim} -> heads {model.embed_dim} "
                  f"(feat_proj {sum(p.numel() for p in model.feat_proj.parameters()):,} params); "
                  f"feature KD is computed on the PROJECTED tokens", flush=True)
        kd_loss = KLDivergenceLoss(student_dim=student_dim,
                                   teacher_dim=teacher_dim,
                                   temperature=args.kd_temperature,
                                   softmax_dim=args.kd_softmax_dim).to(device)
        n_s = sum(p.numel() for p in model.parameters())
        n_t = sum(p.numel() for p in teacher.parameters())
        print(f"[distill] student backbone '{args.backbone}' embed_dim={student_dim} "
              f"({n_s:,} params total) <- teacher embed_dim={teacher_dim} "
              f"({n_t:,} params). compression {n_t/max(n_s,1):.2f}x", flush=True)
        print(f"[distill] lambda_kd={args.lambda_kd} T={args.kd_temperature} "
              f"softmax_dim={args.kd_softmax_dim} "
              f"projector={'yes' if kd_loss.needs_proj else 'no (same width)'}",
              flush=True)
        if student_dim == teacher_dim and getattr(model, 'feat_proj', None) is None:
            print("[distill] WARNING: student and teacher have the SAME width - "
                  "is --backbone actually set to a smaller ViT?", flush=True)
        # the projector is part of what must be optimised
        _kd_params = list(kd_loss.parameters())
        if _kd_params:
            optimizer.add_param_group({'params': _kd_params})

    trainer = Trainer(model=model, loss=loss, optimizer=optimizer, device=device,
                      args=args, scheduler=scheduler, teacher=teacher, kd_loss=kd_loss)

    print()
    print(f"ARGS: {trainer.args}")
    print(f"LOG_DIR: {trainer.args.log_dir}")
    print()

    if args.eval_only:
        # EVALUATION ON AN ANNYONE SPLIT (--eval_split val|test)
        # The AnnyOne loader is built inside the training branch below, so
        # build the requested split here. Index layout (see anny_split_ranges):
        #   [0, val_start) train | [val_start, test_start) val | [test_start, N) test
        # --val_data loaders (e.g. THREEDPW) built above are evaluated as well.
        trainer.eval_checkpoint = args.pretrained
        trainer.eval_epoch = pretrained_epoch
        want_test = args.eval_split == 'test' and args.test_anny_n > 0
        want_val = args.eval_split == 'val' and args.val_anny_n > 0
        if args.train_data == 'AnnyOne' and args.eval_split == 'test' and args.test_anny_n <= 0:
            print("ERROR: --eval_split test needs --test_anny_n > 0.", flush=True)
        if args.train_data == 'AnnyOne' and (want_test or want_val):
            from torch.utils.data import Subset
            eval_dataset = AnnyOne(data_folder='/netscratch/najib/anydataset/',
                                   img_size=args.img_size)
            n_total = len(eval_dataset)
            val_start, test_start = anny_split_ranges(n_total, args.val_anny_n, args.test_anny_n)
            start, end = (test_start, n_total) if want_test else (val_start, test_start)
            print(f"AnnyOne {args.eval_split.upper()} split: {end - start} samples "
                  f"(indices {start}..{end - 1}) out of {n_total} "
                  f"[train 0..{val_start - 1} | val {val_start}..{test_start - 1} | "
                  f"test {test_start}..{n_total - 1}]", flush=True)

            # Did the checkpoint train on any of these indices? Reconstructed
            # from the args saved in the checkpoint (assumes the dataset size
            # has not changed since training).
            train_end = anny_trained_end(n_total, pretrained_args)
            if train_end is None:
                n_seen = ''
                print("  NOTE: cannot tell which AnnyOne indices the checkpoint was "
                      "trained on (no AnnyOne training args in it); overlap NOT checked.",
                      flush=True)
            else:
                n_seen = max(0, min(end, train_end) - start)
                if n_seen > 0:
                    print(f"  WARNING: {n_seen}/{end - start} of these samples "
                          f"(indices {start}..{min(end, train_end) - 1}) WERE SEEN IN "
                          f"TRAINING by this checkpoint (it trained on 0..{train_end - 1}). "
                          f"These numbers are optimistic, not a clean held-out result.",
                          flush=True)
                else:
                    print(f"  OK: disjoint from the checkpoint's training range "
                          f"0..{train_end - 1}.", flush=True)

            def _test_collate(batch):
                try:
                    img, annot = raw_collate(batch)
                except Exception as e:
                    print(f"DATALOADER CRASH: {e}", flush=True)
                    return None, None
                if annot is None:
                    return None, None
                if 'idx' not in annot:
                    annot['idx'] = torch.arange(img.shape[0])
                return img, annot

            eval_subset = Subset(eval_dataset, list(range(start, end)))
            eval_subset.name = 'annyone'
            # tensorboard tag: 'holdout' matches what training logs for val
            eval_subset.split = 'test' if want_test else 'holdout'
            eval_subset.subsample = 1
            eval_subset.results_split = args.eval_split
            eval_subset.n_seen_in_training = n_seen
            l_val_data.append(DataLoader(eval_subset, batch_size=1, shuffle=False,
                                         num_workers=args.num_workers, drop_last=False,
                                         collate_fn=_test_collate))

        if not l_val_data:
            print("ERROR: --eval_only but no evaluation set was built. For AnnyOne "
                  "pass --train_data AnnyOne with --eval_split test --test_anny_n <N> "
                  "or --eval_split val --val_anny_n <N>; for 3DPW pass "
                  "--val_data THREEDPW --val_split test --val_subsample 1 --val_n -1.",
                  flush=True)
        for val_data in l_val_data:
            trainer.evaluate(val_data)
    else:
        # train_data = DataLoader(eval(args.train_data)(split=f"{args.train_split}", 
        #                                           training=1, 
        #                                           img_size=args.img_size,
        #                                           n_iter=args.batch_size * args.n_iters_per_epoch,
        #                                           subsample=args.train_subsample,
        #                                           extension=args.extension,
        #                                           res=args.res,
        #                                           n=args.train_n, # for debugging purpose only
        #                                           crops=args.crops,
        #                                           flip=args.flip,
        #                                           ),
        #                   batch_size=args.batch_size,
        #                   num_workers=args.num_workers,
        #                   shuffle=True,
        #                   drop_last=True,
        #                   collate_fn=collate_fn,
        #                   )
        # --- CUSTOM ANNYONE DATALOADER HACK (V4) ---
        from torch.utils.data import Subset # <-- Add this import!

        # --- CUSTOM ANNYONE DATALOADER (FULL DATASET) ---
        # anny_dataset = AnnyOne(data_folder='/netscratch/najib/anydataset/', img_size=args.img_size)

        # def bulletproof_collate(batch):
        #     img, annot = raw_collate(batch)
        #     if annot is None:
        #         img, annot = raw_collate([anny_dataset[0]] * len(batch))
        #     if 'idx' not in annot:
        #         annot['idx'] = torch.arange(img.shape[0])
        #     return img, annot

        # train_data = DataLoader(anny_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=bulletproof_collate, num_workers=args.num_workers)

        if args.train_data == 'AnnyOne':
            try:
                train_dataset = AnnyOne(
                    data_folder='/netscratch/najib/anydataset/',
                    img_size=args.img_size,
                )
                print(f"AnnyOne dataset size: {len(train_dataset)}", flush=True)
                sample = train_dataset[0]
                print(f"AnnyOne[0] OK — keys: {sample[1].keys() if isinstance(sample, tuple) else 'N/A'}", flush=True)
            except Exception as e:
                import traceback
                print("AnnyOne CRASHED:", flush=True)
                traceback.print_exc()
                raise

            # Subset the dataset if --train_n is set
            full_anny_dataset = train_dataset  # keep the full dataset for the val slice
            if args.train_n > 0:
                from torch.utils.data import Subset
                indices = list(range(min(args.train_n, len(train_dataset))))
                train_dataset = Subset(train_dataset, indices)
                print(f"Using subset of {len(train_dataset)} samples from AnnyOne dataset.", flush=True)
                if args.test_anny_n > 0:
                    _vs, _ = anny_split_ranges(len(full_anny_dataset), args.val_anny_n, args.test_anny_n)
                    if len(train_dataset) > _vs:
                        print(f"WARNING: --train_n {args.train_n} reaches into the val/test "
                              f"tail (starts at index {_vs}); splits are NOT disjoint.", flush=True)
            elif args.test_anny_n > 0:
                # held-out TEST split: train excludes both val and test
                from torch.utils.data import Subset
                _vs, _ts = anny_split_ranges(len(train_dataset), args.val_anny_n, args.test_anny_n)
                train_dataset = Subset(train_dataset, list(range(_vs)))
                print(f"Training on {_vs} samples; {_ts - _vs} reserved for val "
                      f"(indices {_vs}..{_ts - 1}) and {len(full_anny_dataset) - _ts} "
                      f"for test (indices {_ts}..{len(full_anny_dataset) - 1}, never "
                      f"evaluated during training).", flush=True)
            elif args.val_anny_n > 0:
                # full-dataset training: hold out the LAST val_anny_n samples
                from torch.utils.data import Subset
                keep = len(train_dataset) - args.val_anny_n
                train_dataset = Subset(train_dataset, list(range(keep)))
                print(f"Training on {keep} samples; last {args.val_anny_n} reserved for holdout.", flush=True)

            # Define it RIGHT HERE, after train_dataset exists
            def bulletproof_collate(batch):
                try:
                    img, annot = raw_collate(batch)
                except Exception as e:
                    print(f"CRITICAL DATALOADER CRASH: {e}", flush=True) # <-- ADD THIS
                    return None, None
                if annot is None:
                    return None, None
                if 'idx' not in annot:
                    annot['idx'] = torch.arange(img.shape[0])
                return img, annot

            train_data = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=bulletproof_collate,
                num_workers=args.num_workers,
                drop_last=True,
            )

            # Held-out AnnyOne validation slice (enabled with --val_anny_n > 0).
            # Taken from the END of the dataset. When --train_n > 0 the train
            # subset is [0, train_n) so the tail never overlaps; when training
            # on the full dataset the last val_anny_n samples are excluded
            # from training below to keep the holdout clean.
            if args.val_anny_n > 0:
                from torch.utils.data import Subset
                if args.test_anny_n > 0:
                    # val = the val_anny_n samples just BEFORE the test tail
                    val_start, val_end = anny_split_ranges(
                        len(full_anny_dataset), args.val_anny_n, args.test_anny_n)
                else:
                    val_start = max(0, len(full_anny_dataset) - args.val_anny_n)
                    val_end = len(full_anny_dataset)
                if val_end > val_start:
                    val_subset = Subset(full_anny_dataset, list(range(val_start, val_end)))
                    # attrs read by evaluate() for logging/tensorboard tags
                    val_subset.name = 'annyone'
                    val_subset.split = 'holdout'
                    val_subset.subsample = 1
                    val_subset.results_split = 'val'
                    val_loader = DataLoader(
                        val_subset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        drop_last=False,
                        collate_fn=bulletproof_collate,
                    )
                    l_val_data.append(val_loader)
                    print(f"AnnyOne holdout val: {len(val_subset)} samples "
                          f"(indices {val_start}..{val_end - 1})", flush=True)
                else:
                    print("WARNING: --val_anny_n requested but no samples left after the train subset", flush=True)

        elif args.train_data == 'BEDLAM':
            train_dataset = BEDLAM(
                split=args.train_split,
                training=1,
                img_size=args.img_size,
                n_iter=args.batch_size * args.n_iters_per_epoch,
                subsample=args.train_subsample,
                n=args.train_n,
            )
            train_data = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn,
            )

        else:
            raise ValueError(f"Unknown --train_data: {args.train_data}. Choose 'AnnyOne' or 'BEDLAM'.")

        trainer.fit(train_data, l_val_data)
        
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--train_data', type=str, default='BEDLAM')
    parser.add_argument('--train_split', type=str, default='training')
    parser.add_argument('--train_n', type=int, default=0)
    parser.add_argument('--val_data', type=str, nargs='+', default=[])
    parser.add_argument('--val_split', type=str, nargs='+', default=[])
    parser.add_argument('--val_n', type=int, nargs='+', default=[])
    parser.add_argument('--val_subsample', type=int, nargs='+', default=[])
    parser.add_argument('--save_dir', type=str, default='logs')
    parser.add_argument('--name', type=str, default='trainval')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--pretrained_remap', type=int, default=1, choices=[0, 1],
                        help="remap 'encoder.backbone.*' -> 'backbone.encoder.*' so a Multi_HMR "
                             "checkpoint (e.g. multiHMR_672_L_anny) loads into this Model. "
                             "Harmless no-op for checkpoints already trained with this script.")
    parser.add_argument('--load_only_backbone', type=int, default=0, choices=[0, 1],
                        help='keep only backbone.* weights from --pretrained; all heads start '
                             'from fresh init (matches the original repo flag of the same name)')
    parser.add_argument('--freeze_backbone', type=int, default=0, choices=[0, 1],
                        help='freeze the backbone after loading: fixed feature extractor, '
                             'only the heads train (task step 2)')
    parser.add_argument('--unfreeze_last_n_blocks', type=int, default=0,
                        help='with --freeze_backbone 1: leave the LAST N transformer blocks of the '
                             'ViT (plus its final norm) trainable. Early blocks learn generic '
                             'features that transfer as-is; later blocks are task-specific and '
                             'benefit from adaptation. ViT-L has 24 blocks; 4-6 is a good range. '
                             '0 = freeze everything (original step-2 behavior).')
    parser.add_argument('--mask_helper_joints_train', type=int, default=0, choices=[0, 1],
                        help='force the 88 Anny helper/deform bones (breast, spine/neck subdivisions, '
                             'face micro-bones) to identity during TRAINING, and mask the matching GT '
                             'rotation targets, so the rotmat loss concentrates on the 75 learnable '
                             'joints. Mirrors the original repo\'s useful_rotmat design.')
    parser.add_argument('--min_backbone_match', type=float, default=90.0,
                        help='abort if fewer than this %% of backbone tensors were initialised '
                             'from --pretrained. Guards against silently training a random ViT. '
                             'Set to 0 to disable.')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--train_subsample', type=int, default=1)
    parser.add_argument('--num_workers', '-j', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=336)
    parser.add_argument('--backbone', type=str, default='dinov2_vits14', choices=['dinov2_vitl14', 'dinov2_vitb14', 'dinov2_vits14'])
    parser.add_argument("--n_iters_per_epoch", "-iter", type=int, default=100)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--nb_max_ckpt", type=int, default=10)    
    parser.add_argument('--amp', type=int, default=1, choices=[0,1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument('--use_efficient_attention', type=int, default=1, choices=[0,1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-6, help='learning rate (absolute lr)')
    parser.add_argument('--lr_decay_every', type=int, default=30,
                        help='decay learning rate every N epochs (StepLR). Important for '
                             'resumed runs: a constant LR tuned for cold-start training can '
                             'be too aggressive on a partially-trained model and cause '
                             'NaN/Inf gradients (observed on the last resume without decay).')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5,
                        help='multiply LR by this factor every --lr_decay_every epochs')
    parser.add_argument('--eval_only', type=int, default=0, choices=[0,1])
    # ---- knowledge distillation ----
    parser.add_argument('--distill_teacher_ckpt', type=str, default='',
                        help='path to a trained checkpoint used as the FROZEN TEACHER. '
                             'Setting this turns the run into student training: --backbone '
                             'then specifies the STUDENT (e.g. dinov2_vits14).')
    parser.add_argument('--lambda_kd', type=float, default=1.0,
                        help='weight of the KL distillation term added to the task loss')
    parser.add_argument('--kd_temperature', type=float, default=4.0,
                        help='softmax temperature for distillation. Higher = softer targets. '
                             'Loss is scaled by T^2 so gradient magnitude stays comparable.')
    parser.add_argument('--init_heads_from_teacher', type=int, default=0, choices=[0, 1],
                        help='copy every non-backbone tensor whose name AND shape match from the '
                             'teacher into the student before training (HPH, pose/shape/depth '
                             'MLPs). Layers touching embed_dim stay random and are listed.')
    parser.add_argument('--head_dim', type=int, default=0,
                        help='student only: width the heads are built at, with a linear projector '
                             'after the backbone. Set to the TEACHER width (1024 for ViT-L) so every '
                             'head tensor transfers with --init_heads_from_teacher. 0 = off.')
    parser.add_argument('--freeze_heads_epochs', type=int, default=0,
                        help='with --init_heads_from_teacher: keep every tensor copied from the '
                             'teacher FROZEN for this many epochs so the random embed_dim '
                             'projections learn to feed them first. 0 = off.')
    parser.add_argument('--lambda_kd_out', type=float, default=0.0,
                        help='weight of OUTPUT-level distillation: L1 between student and teacher '
                             'predicted rotmat/shape/depth on the same image. 0 = off.')
    parser.add_argument('--kd_softmax_dim', type=str, default='channel',
                        choices=['channel', 'token'],
                        help="'channel': each patch token becomes a distribution over feature "
                             "channels (standard). 'token': distribution over spatial tokens.")
    parser.add_argument('--test_anny_n', type=int, default=0,
                        help='size of the held-out AnnyOne TEST split = the LAST N samples '
                             '(0 = no test split). When > 0, the --val_anny_n validation '
                             'samples are the ones immediately BEFORE the test split, and '
                             'training excludes both. Use --eval_only --eval_split test to '
                             'evaluate it.')
    parser.add_argument('--person_center', type=str, default='head', choices=['pelvis', 'head', 'nose'])
    parser.add_argument('--visu_to_save', type=int, default=0)
    parser.add_argument('--extension', type=str, default='png', choices=['png', 'jpg'])
    parser.add_argument('--res', type=int, default=None, choices=[None, 512, 1280])
    parser.add_argument('--num_betas', type=int, default=11, choices=[10, 11])
    parser.add_argument('--use_anny_shape', type=int, default=0, choices=[0,1], help='use anny_shape from the dataset as GT phenotypes instead of zeros')
    parser.add_argument('--val_anny_n', type=int, default=0, help='number of held-out AnnyOne samples for validation (0=disabled); taken from the END of the dataset (the last N samples, or the N samples just before the --test_anny_n test split) and excluded from training unless --train_n is set')
    parser.add_argument('--eval_split', type=str, default='val', choices=['val', 'test'],
                        help='with --eval_only and --train_data AnnyOne: which AnnyOne split to '
                             'evaluate (val needs --val_anny_n > 0, test needs --test_anny_n > 0)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='where evaluate() appends <run_name>.csv and all_results.jsonl '
                             '(relative paths are resolved against the repo directory)')
    parser.add_argument('--eval_freq', type=int, default=1, help='run evaluation every N epochs')
    parser.add_argument('--det_thresh', type=float, default=0.2)
    parser.add_argument('--nms_kernel_size', type=int, default=3)
    parser.add_argument('--crops', type=int, nargs='+', default=[0])
    parser.add_argument('--flip', type=int, default=1, choices=[0,1])
    parser.add_argument('--brightness', type=float, default=0.)
    parser.add_argument('--contrast', type=float, default=0.)
    parser.add_argument('--saturation', type=float, default=0.)
    parser.add_argument('--hue', type=float, default=0.)

    parser = Loss.add_specific_args(parser)
    args = parser.parse_args()
    args.max_epochs = args.max_iter // args.n_iters_per_epoch

    main(args)