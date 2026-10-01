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

class Trainer(object):
    def __init__(self, model, loss, optimizer, device, args, best_val=1e5):
        self.model = model
        self.loss = loss
        self.device = device
        self.args = args
        self.optimizer = optimizer
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
            
            target['rotmat'] = anny_rotmats_ortho.reshape(nhv, 163, 3, 3)   
            target['rotvec'] = roma.rotmat_to_rotvec(
                anny_rotmats_ortho
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
        print(f"\nTRAIN: ")
        self.model.train()

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

    @torch.no_grad()
    def evaluate(self, data):
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

        for i, (x,y) in enumerate(tqdm(data)):
            if x is None or y is None:
                continue
            # move tensor to device
            y = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in y.items()}
            
            # preprare gt by computing mesh and 3d/2d joints
            gt = self.prepare_gt(y=y)
            if gt is None:
                continue

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
        return meters['pve'].avg
    
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(pretrained_backbone=1, **vars(args))
    model = model.to(device)

    # Load from a pretrained model
    if args.pretrained is not None and os.path.isfile(args.pretrained):
        print(f"Loading weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained)
        log = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"{log}")

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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss = Loss(args)

    trainer = Trainer(model=model, loss=loss, optimizer=optimizer, device=device, args=args)

    print()
    print(f"ARGS: {trainer.args}")
    print(f"LOG_DIR: {trainer.args.log_dir}")
    print()

    if args.eval_only:
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
                val_start = max(0, len(full_anny_dataset) - args.val_anny_n)
                val_end = len(full_anny_dataset)
                if val_end > val_start:
                    val_subset = Subset(full_anny_dataset, list(range(val_start, val_end)))
                    # attrs read by evaluate() for logging/tensorboard tags
                    val_subset.name = 'annyone'
                    val_subset.split = 'holdout'
                    val_subset.subsample = 1
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
    parser.add_argument('--eval_only', type=int, default=0, choices=[0,1])
    parser.add_argument('--person_center', type=str, default='head', choices=['pelvis', 'head', 'nose'])
    parser.add_argument('--visu_to_save', type=int, default=0)
    parser.add_argument('--extension', type=str, default='png', choices=['png', 'jpg'])
    parser.add_argument('--res', type=int, default=None, choices=[None, 512, 1280])
    parser.add_argument('--num_betas', type=int, default=11, choices=[10, 11])
    parser.add_argument('--use_anny_shape', type=int, default=0, choices=[0,1], help='use anny_shape from the dataset as GT phenotypes instead of zeros')
    parser.add_argument('--val_anny_n', type=int, default=0, help='number of held-out AnnyOne samples for validation (0=disabled); taken from indices after --train_n')
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