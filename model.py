# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

from torch import nn
import torch
import numpy as np
import roma
import anny
# import copy

from utils import (
    unpatch,
    inverse_perspective_projection,
    perspective_projection,        # ← ADD this if not already imported
    undo_focal_length_normalization,
    undo_log_depth,
)
from utils import rotation_to_homogeneous 

from blocks import (
    Dinov2Backbone,
    FourierPositionEncoding,
    TransformerDecoder,
)
from utils import rot6d_to_rotmat, rebatch, pad_to_max
import torch.nn as nn
import numpy as np
import einops
from utils.constants import MEAN_PARAMS

# Mapping from 163 Anny joints -> 55 SMPL-X joints
# Used to run the existing SMPL-X layer for mesh generation
# ANNY_TO_SMPLX_MAPPING = [
#     0, 2, 22, 47, 4, 24, 44, 6, 26, 43, 7, 27, 100, 48, 74, 103,
#     50, 76, 52, 78, 54, 80, 104, 143, 148, 59, 60, 61, 63, 64, 65,
#     71, 72, 73, 67, 68, 69, 55, 56, 57, 85, 86, 87, 89, 90, 91,
#     97, 98, 99, 93, 94, 95, 81, 82, 83
# ]

# smpl_layer joint order: root, body(21), lhand(15), rhand(15), jaw — no leye/reye
# SMPLX_TO_SMPL_LAYER_INDICES = list(range(23)) + list(range(25, 55))  # 53 joints

class Model(nn.Module):
    """A ViT backbone followed by a "HPH" head (stack of cross attention layers with queries corresponding to detected humans.)"""

    def __init__(
        self,
        backbone="dinov2_vitb14",
        pretrained_backbone=False,
        img_size=896,
        camera_embedding="geometric",
        camera_embedding_num_bands=16,
        camera_embedding_max_resolution=64,
        nearness=True,
        xat_depth=2,
        xat_num_heads=8,
        dict_smpl_layer=None,
        person_center="head",
        clip_dist=True,
        num_betas=11,
        head_dim=None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.img_size = img_size
        self.nearness = nearness
        self.clip_dist = (clip_dist,)
        self.xat_depth = xat_depth
        self.xat_num_heads = xat_num_heads
        self.num_betas = num_betas

        self.backbone = Dinov2Backbone(backbone, pretrained=pretrained_backbone)
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        # --- projector-fed heads (see apply_student_fix_v3.py) ---------------
        # With head_dim set (student distillation), a linear layer maps the
        # backbone tokens to head_dim and EVERY head is built at that width, so
        # a teacher of that width transfers its heads completely. None/0 keeps
        # the heads at the backbone width (original behaviour).
        self.backbone_dim = self.backbone.embed_dim
        self.feat_proj = None
        if head_dim is not None and int(head_dim) > 0 and int(head_dim) != self.backbone_dim:
            self.feat_proj = nn.Linear(self.backbone_dim, int(head_dim))
            self.embed_dim = int(head_dim)
        assert self.img_size % self.patch_size == 0, "Invalid img size"

        self.fovn = 60
        self.camera_embedding = camera_embedding
        self.camera_embed_dim = 0
        if self.camera_embedding is not None:
            if not self.camera_embedding == "geometric":
                raise NotImplementedError(
                    "Only geometric camera embedding is implemented"
                )
            self.camera = FourierPositionEncoding(
                n=3,
                num_bands=camera_embedding_num_bands,
                max_resolution=camera_embedding_max_resolution,
            )
            self.camera_embed_dim = self.camera.channels

        self.mlp_classif = regression_mlp(
            [self.embed_dim, self.embed_dim, 1]
        )
        self.mlp_offset = regression_mlp([self.embed_dim, self.embed_dim, 2])

        
        self.nrot = 163                          # Anny has 163 joints
        self.num_betas = 11                      # Always 11 for Anny
        self.person_center_name = person_center

        self.body_model = anny.create_fullbody_model(
            remove_unattached_vertices=False,
            all_phenotypes=True,
        ).float()

        joint_names = list(self.body_model.bone_labels)
        self.person_center_idx = joint_names.index(person_center)
        self.body_model.set_skinning_method('lbs')
        self.body_model.name = 'anny'

        # --- HELPER-BONE MASK (chest/neck deformation fix) ---
        # The original repo's Multi_HMR forces ~78 of the 163 bones to identity
        # (rest pose) via this mask: Anny's rig has helper/deform bones (breast,
        # spine/neck subdivisions, face micro-bones) that a network cannot
        # predict reliably from pixels. This Model trained WITHOUT the mask, so
        # those bones carry noisy rotations -> visible chest crease / pinched
        # neck at inference even when the limb pose is correct.
        # Mask copied verbatim from multi_hmr_anny/multi_hmr.py (1 = keep the
        # predicted rotation, 0 = replace with identity/rest).
        # persistent=False keeps it out of state_dict so old checkpoints load
        # at 100% coverage unchanged.
        _useful = torch.Tensor([1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
                                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0.]).unsqueeze(0)
        self.register_buffer('useful_rotmat', _useful, persistent=False)

        # Inference-time diagnostic toggles (set by demo.py from CLI flags;
        # both default OFF so training behavior is completely unchanged):
        #   mask_helper_joints -> apply the mask above to predicted rotations
        #   neutral_shape      -> force all 11 phenotypes to 0.5 (isolates
        #                         whether deformation comes from shape values)
        self.mask_helper_joints = False
        # Training-side version of the same mask (set by train.py from
        # --mask_helper_joints_train): forces helper-bone predictions to
        # identity DURING TRAINING as well, so together with the matching GT
        # masking in prepare_gt the rotmat loss budget concentrates entirely
        # on the 75 learnable joints (incl. the cervical neck chain).
        self.mask_helper_joints_train = False
        self.neutral_shape = False

        # --- HAND RELAXATION (name-based, robust to index changes) ---
        # Finger bones are among the 75 KEPT joints but occupy only a few
        # pixels at 672px input, so their predictions are undertrained and
        # render as claw-like hands. relax_hands=True forces them to rest
        # pose at inference. Bones are found by NAME so no index guessing:
        # anything containing 'finger' or 'thumb' in the rig.
        _hand = torch.zeros(1, len(joint_names))
        _n_hand = 0
        for _ji, _jn in enumerate(joint_names):
            if ('finger' in _jn.lower()) or ('thumb' in _jn.lower()):
                _hand[0, _ji] = 1.0
                _n_hand += 1
        self.register_buffer('hand_bone_mask', _hand, persistent=False)
        print(f"[model] hand_bone_mask: {_n_hand} finger/thumb bones identified by name")
        self.relax_hands = False

        # --- SELECTIVE PHENOTYPE NEUTRALIZATION ---
        # List of phenotype names (e.g. ['proportions', 'age']) forced to the
        # neutral value 0.5 at inference, keeping all others as predicted.
        # Motivated by a systematic bias observed on real photos: the shape
        # head outputs proportions~0.8 / age~0.74 for EVERY person, a learned
        # out-of-domain offset that distorts neck/head proportions.
        self.neutral_phenotype_names = []

        # --- REST POSE DIAGNOSTIC ---
        # rest_pose=True forces every joint EXCEPT the root (index 0) to
        # identity at inference, so the mesh renders in Anny's neutral
        # standing pose with the predicted shape + placement. Use to separate
        # pose-driven deformation (disappears here) from shape/body-model-
        # driven deformation (persists here). Root is kept so the body still
        # faces the right way.
        self.rest_pose = False

        # eye matrix used for pose init (same as multi_hmr-3.py)
        self.eye = nn.Parameter(torch.eye(3).unsqueeze(0), requires_grad=False)

        self.x_attention_head = HPH(
            num_body_joints=self.nrot,          # 163, not 163-1
            context_dim=self.embed_dim + self.camera_embed_dim,
            dim=1024,
            depth=self.xat_depth,
            heads=self.xat_num_heads,
            mlp_dim=1024,
            dim_head=32,
            dropout=0.0,
            emb_dropout=0.0,
            at_token_res=self.img_size // self.patch_size,
            num_betas=11,                       # hardcode 11, not from args
        )

    def detection(self, z, nms_kernel_size, det_thresh, N, idx=None, is_training=False):
        """Detection score on the entire low res image"""
        scores = _sigmoid(self.mlp_classif(z))
        scores = unpatch(
            scores, patch_size=1, c=scores.shape[2], img_size=int(np.sqrt(N))
        )

        if not is_training:
            if nms_kernel_size > 1:
                scores = _nms(scores, kernel=nms_kernel_size)
            _scores = torch.permute(scores, (0, 2, 3, 1))
            idx = apply_threshold(det_thresh, _scores)
        else:
            assert idx is not None

        scores_detected = scores[idx[0], idx[3], idx[1], idx[2]]
        scores = torch.permute(scores, (0, 2, 3, 1))
        return scores, scores_detected, idx

    def embedd_camera(self, K, z):
        """Embed viewing directions using fourrier encoding."""
        bs = z.shape[0]
        _h, _w = list(z.shape[-2:])
        points = (
            torch.stack(
                [
                    torch.arange(0, _h, 1).reshape(-1, 1).repeat(1, _w),
                    torch.arange(0, _w, 1).reshape(1, -1).repeat(_h, 1),
                ],
                -1,
            )
            .to(z.device)
            .float()
        )
        points = points * self.patch_size + self.patch_size // 2
        points = points.reshape(1, -1, 2).repeat(bs, 1, 1)
        distance = torch.ones(bs, points.shape[1], 1).to(K.device)
        rays = inverse_perspective_projection(points, K, distance)
        rays_embeddings = self.camera(pos=rays)
        z_K = rays_embeddings.reshape(bs, _h, _w, self.camera_embed_dim)
        return z_K

    def to_euclidean_dist(self, x, dist, _K):
        focal = _K[:, [0], [0]]
        dist = undo_focal_length_normalization(
            dist, focal, fovn=self.fovn, img_size=x.shape[-1]
        )
        if self.nearness:
            dist = undo_log_depth(dist)
        if self.clip_dist:
            dist = torch.clamp(dist, 0, 50)
        return dist

    def forward(
        self,
        x,
        idx=None,
        det_thresh=0.3,
        nms_kernel_size=3,
        K=None,
        is_training=False,
        *args,
        **kwargs,
    ):
        persons = []
        out = {}

        # Feature extraction
        z = self.backbone(x)
        if self.feat_proj is not None:
            out['backbone_feat_raw'] = z          # [B, N, backbone_dim], for reference
            z = self.feat_proj(z)                 # [B, N, head_dim] - what the heads see
        B, N, C = z.size()

        # --- KNOWLEDGE DISTILLATION TAP ---
        # Expose the raw backbone patch tokens [B, N, C] here: immediately
        # after patch decoding and BEFORE any projection head (detection MLP,
        # HPH, pose/shape/dist MLPs). train.py reads out['backbone_feat'] to
        # compute the KL divergence against the frozen teacher's tokens, so the
        # student matches the REPRESENTATION rather than only the predictions.
        out['backbone_feat'] = z

        # Detection
        scores, scores_det, idx = self.detection(
            z,
            nms_kernel_size=nms_kernel_size,
            det_thresh=det_thresh,
            N=N,
            idx=idx,
            is_training=is_training,
        )
        if len(idx[0]) == 0 and not is_training:
            return persons

        # Map of Dense Feature
        z = unpatch(
            z, patch_size=1, c=z.shape[2], img_size=int(np.sqrt(N))
        )
        z_all = z

        # Extract the 'central' features
        z = torch.reshape(
            z, (z.shape[0], 1, z.shape[1] // 1, z.shape[2], z.shape[3])
        )
        z_central = z[idx[0], idx[3], :, idx[1], idx[2]]

        # 2D offset regression
        offset = self.mlp_offset(z_central)

        # Camera intrinsics
        K_det = K[idx[0]]
        z_K = self.embedd_camera(K, z)
        z_central = torch.cat(
            [z_central, z_K[idx[0], idx[1], idx[2]]], 1
        )
        z_all = torch.cat(
            [z_all, z_K.permute(0, 3, 1, 2)], 1
        )
        z = torch.cat([z, z_K.permute(0, 3, 1, 2).unsqueeze(1)], 2)

        # Distance for estimating the 3D location in 3D space
        loc = torch.stack([idx[2], idx[1]]).permute(1, 0)
        loc = (loc + 0.5 + offset) * self.patch_size

        # ANNY parameter regression
        kv = z_all[idx[0]]
        pred_anny_params, pred_cam = self.x_attention_head(
            z_central, kv, idx_0=idx[0], idx_det=idx
        )

        # Get outputs from the Anny head
        shape = pred_anny_params["shape"]
        rotmat = pred_anny_params["rotmat"]

        # --- CHEST/NECK FIX: replace helper-bone rotations with identity ---
        # (inference-time toggle, see __init__; mirrors the original repo's
        # useful_rotmat masking). Applied BEFORE rotvec/homogeneous conversion
        # so every downstream output (rotvec, mesh, projections) is consistent.
        if (self.mask_helper_joints and not is_training) or (self.mask_helper_joints_train and is_training):
            m = self.useful_rotmat.reshape(1, -1, 1, 1).to(rotmat.dtype)
            eye = torch.eye(3, device=rotmat.device, dtype=rotmat.dtype).reshape(1, 1, 3, 3)
            rotmat = m * rotmat + (1.0 - m) * eye

        # --- HAND RELAXATION: force finger/thumb bones to rest at inference ---
        # (see __init__; fingers are trainable but undertrained at 672px, so
        # this trades claw-like predicted fingers for clean neutral hands).
        if self.relax_hands and not is_training:
            hm = self.hand_bone_mask.reshape(1, -1, 1, 1).to(rotmat.dtype)
            eye = torch.eye(3, device=rotmat.device, dtype=rotmat.dtype).reshape(1, 1, 3, 3)
            rotmat = (1.0 - hm) * rotmat + hm * eye

        # --- REST POSE DIAGNOSTIC: identity on every joint except the root ---
        if self.rest_pose and not is_training:
            eye = torch.eye(3, device=rotmat.device, dtype=rotmat.dtype).reshape(1, 1, 3, 3)
            rest = eye.expand(rotmat.shape[0], rotmat.shape[1], 3, 3).clone()
            rest[:, 0] = rotmat[:, 0]   # keep predicted root orientation
            rotmat = rest

        rotvec = roma.rotmat_to_rotvec(rotmat.reshape(-1, 3, 3)).reshape(-1, 163, 3)

        # Compute depth from predicted camera params (pred_cam[:, 0] is the raw dist)
        dist_postprocessed = pred_cam[:, 0]                          # [N] — normalized, for loss
        dist = self.to_euclidean_dist(x, pred_cam[:, 0:1], K_det)   # [N, 1] — meters, for 3D
        dist = dist.squeeze(1)                                        # [N]

        # --- NATIVE ANNY MESH GENERATION ---

        # 1. Shape: apply sigmoid (Anny phenotypes are in [0,1])
        shape = torch.sigmoid(shape)  # [N, 11]
        # Diagnostic toggle: force neutral phenotypes to isolate whether a
        # deformation comes from predicted shape values or from pose.
        if self.neutral_shape and not is_training:
            shape = torch.full_like(shape, 0.5)
        # Selective version: neutralize only the phenotypes listed by name in
        # self.neutral_phenotype_names (e.g. ['proportions', 'age']), keeping
        # all other predictions. Targets the systematic out-of-domain bias
        # without discarding per-person gender/weight/etc.
        if self.neutral_phenotype_names and not is_training:
            labels = list(self.body_model.phenotype_labels)
            for name in self.neutral_phenotype_names:
                if name in labels:
                    shape = shape.clone()
                    shape[:, labels.index(name)] = 0.5
                else:
                    print(f"[model] WARNING: unknown phenotype '{name}' "
                          f"(valid: {labels})")
        # Forward ALL predicted phenotypes into the body model. The dataset's
        # anny_shape has 11 real phenotypes (gender, age, muscle, weight,
        # height, proportions, cupsize, firmness, african, asian, caucasian)
        # and the shape head predicts all 11 — dropping any of them here would
        # make the rendered/eval mesh ignore predicted body-shape variation.
        _shape = {k: shape[:, l] for l, k in enumerate(self.body_model.phenotype_labels)}

        # 2. Convert rotmat [N, 163, 3, 3] -> homogeneous [N, 163, 4, 4]
        rotmat_homo = rotation_to_homogeneous(rotmat)  # uses util from multi_hmr-3.py

        # 3. Run Anny body model
        anny_output = self.body_model(pose_parameters=rotmat_homo, phenotype_kwargs=_shape)
        v3d = anny_output['vertices']              # [N, V, 3]
        j3d = anny_output['bone_poses'][:, :, :3, -1]  # [N, 163, 3]

        # 4. Compute 3D translation: unproject 2D loc + predicted depth
        K_det = K[idx[0]]
        transl = inverse_perspective_projection(
            loc.unsqueeze(1), K_det, dist.unsqueeze(1)
        )[:, 0]  # [N, 3]

        # 5. Center mesh on person_center joint, then translate to world
        person_center_j3d = j3d[:, [self.person_center_idx]]   # [N, 1, 3]
        v3d = v3d - person_center_j3d + transl.unsqueeze(1)    # [N, V, 3]
        j3d = j3d - person_center_j3d + transl.unsqueeze(1)    # [N, 163, 3]

        # 6. Project to 2D
        j2d = perspective_projection(j3d, K_det)               # [N, 163, 2]
        v2d = perspective_projection(v3d, K_det)               # [N, V, 2]   
        # 7. Store
        out['v3d'] = v3d
        out['j3d'] = j3d
        out['j2d'] = j2d
        out['v2d'] = v2d              # ← ADD THIS
        out['transl'] = j3d[:, self.person_center_idx]         # [N, 3]
        out['transl_pelvis'] = j3d[:, 0]                       # [N, 3]

        # -------------------------------------------------------------------------

        # Populate output dictionary
        out.update(
            {
                "scores": scores,
                "offset": offset,
                "dist": dist,
                "dist_postprocessed": dist_postprocessed,   # ← ADD THIS
                "rotmat": rotmat,
                "shape": shape,
                "rotvec": rotvec,
                "loc": loc,
                # DO NOT re-add transl / transl_pelvis here — already set above
            }
        )

        assert (
            rotvec.shape[0] == shape.shape[0] == loc.shape[0] == dist.shape[0]
        ), "Incoherent shapes"

        if is_training:
            return out
        else:
            for i in range(idx[0].shape[0]):
                person = {
                    "scores": scores_det[i],
                    "loc": out["loc"][i],
                    "transl": out["transl"][i],
                    "transl_pelvis": out["transl_pelvis"][i],
                    "rotvec": out["rotvec"][i],
                    "shape": out["shape"][i],
                    "v3d": out["v3d"][i],
                    "j3d": out["j3d"][i],
                    "j2d": out["j2d"][i],
                }
                persons.append(person)

            return persons


class HPH(nn.Module):
    """Cross-attention based SMPL Transformer decoder"""

    def __init__(
        self,
        num_body_joints=52,
        context_dim=1280,
        dim=1024,
        depth=2,
        heads=8,
        mlp_dim=1024,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        at_token_res=32,
        num_betas=11,
    ):
        super().__init__()

        self.joint_rep_type, self.joint_rep_dim = "6d", 6

        self.num_anny_joints = 163
        self.nrot = 163

        npose = self.joint_rep_dim * self.num_anny_joints
        self.npose = npose

        self.depth = (depth,)
        self.heads = (heads,)
        self.res = at_token_res
        self.input_is_mean_shape = True
        _context_dim = context_dim
        self.num_betas = num_betas
        assert num_betas in [10, 11]

        transformer_args = dict(
            num_tokens=1,
            token_dim=(
                (npose + self.num_betas + 3 + _context_dim)
                if self.input_is_mean_shape
                else 1
            ),
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            context_dim=context_dim,
        )
        self.transformer = TransformerDecoder(**transformer_args)

        dim = transformer_args["dim"]

        self.npose = self.num_anny_joints * 6

        self.decpose = nn.Linear(dim, 163 * 6)
        self.decshape = nn.Linear(dim, 11)
        self.deccam = nn.Linear(dim, 3)
        self.decexpression = None

        self.set_smpl_init()
        self.init_learned_queries(context_dim)

    def init_learned_queries(self, context_dim, std=0.2):
        self.cross_queries_x = nn.Parameter(torch.zeros(self.res, context_dim))
        torch.nn.init.normal_(self.cross_queries_x, std=std)

        self.cross_queries_y = nn.Parameter(torch.zeros(self.res, context_dim))
        torch.nn.init.normal_(self.cross_queries_y, std=std)

        self.cross_values_x = nn.Parameter(torch.zeros(self.res, context_dim))
        torch.nn.init.normal_(self.cross_values_x, std=std)

        self.cross_values_y = nn.Parameter(torch.zeros(self.res, context_dim))
        torch.nn.init.normal_(self.cross_values_y, std=std)

    def set_smpl_init(self):
        init_body_pose = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).repeat(163).unsqueeze(0)
        init_betas = torch.zeros(1, 11)
        init_cam = torch.tensor([[1.0, 0.0, 0.0]])
        # init_expression = torch.zeros(1, 10)

        self.register_buffer("init_body_pose", init_body_pose)
        self.register_buffer("init_betas", init_betas)
        self.register_buffer("init_cam", init_cam)
        # self.register_buffer("init_expression", init_expression)

    def cross_attn_inputs(self, x, x_central, idx_0, idx_det):
        h, w = x.shape[2], x.shape[3]
        x = einops.rearrange(x, "b c h w -> b (h w) c")

        assert idx_0 is not None, "Learned cross queries only work with multicross"

        if idx_0.shape[0] > 0:
            counts, idx_det_0 = rebatch(idx_0, idx_det)
            old_shape = x_central.shape

            assert idx_det is not None, "idx_det needed for learned_attention"

            xx = einops.rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            queries_xy = (
                self.cross_queries_x[idx_det[1]] + self.cross_queries_y[idx_det[2]]
            )
            x_central = x_central + queries_xy
            assert x_central.shape == old_shape, "Problem with shape"

            x_central, mask = pad_to_max(x_central, counts)

            xx = xx[torch.cumsum(counts, dim=0) - 1]

            values_xy = (
                self.cross_values_x[idx_det[1]] + self.cross_values_y[idx_det[2]]
            )
            xx[idx_det_0, :, idx_det[1], idx_det[2]] += values_xy

            x = einops.rearrange(xx, "b c h w -> b (h w) c")
            num_ppl = x_central.shape[1]
        else:
            mask = None
            num_ppl = 1
            counts = None
        return x, x_central, mask, num_ppl, counts

    def forward(self, x_central, x, idx_0=None, idx_det=None, **kwargs):
        batch_size = x.shape[0]

        x, x_central, mask, num_ppl, counts = self.cross_attn_inputs(
            x, x_central, idx_0, idx_det
        )

        bs = x_central.shape[0] if idx_0.shape[0] else batch_size
        expand = lambda x: x.expand(bs, num_ppl, -1)
        pred_body_pose, pred_betas, pred_cam = [
            expand(x)
            for x in [
                self.init_body_pose,
                self.init_betas,
                self.init_cam,
            ]
        ]
        token = torch.cat([x_central, pred_body_pose, pred_betas, pred_cam], dim=-1)
        if len(token.shape) == 2:
            token = token[:, None, :]

        token_out = self.transformer(token, context=x, mask=mask)

        if mask is not None:
            token_out_list = [token_out[i, :c, ...] for i, c in enumerate(counts)]
            token_out = torch.concat(token_out_list, dim=0)
        else:
            token_out = token_out.squeeze(1)

        reshape = (
            (lambda x: x)
            if idx_0.shape[0] == 0
            else (lambda x: x[0, 0, ...][None, ...])
        )

        pred_body_pose = self.decpose(token_out) + reshape(pred_body_pose)
        pred_betas = self.decshape(token_out) + reshape(pred_betas)
        pred_cam = self.deccam(token_out) + reshape(pred_cam)

        pred_body_pose = pred_body_pose.reshape(-1, 6)
        pred_body_pose = rot6d_to_rotmat(pred_body_pose)
        pred_body_pose = pred_body_pose.view(-1, 163, 3, 3)

        pred_anny_params = {
            "rotmat": pred_body_pose,
            "shape": pred_betas,
        }
        return pred_anny_params, pred_cam


def regression_mlp(layers_sizes):
    assert len(layers_sizes) >= 2
    in_features = layers_sizes[0]
    layers = []
    for i in range(1, len(layers_sizes) - 1):
        out_features = layers_sizes[i]
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    layers.append(torch.nn.Linear(in_features, layers_sizes[-1]))
    return torch.nn.Sequential(*layers)


def apply_threshold(det_thresh, _scores):
    if isinstance(det_thresh, list):
        det_thresh = det_thresh[0]
    idx = torch.where(_scores >= det_thresh)
    return idx


def _nms(heat, kernel=3):
    if kernel not in [2, 4]:
        pad = (kernel - 1) // 2
    else:
        if kernel == 2:
            pad = 1
        else:
            pad = 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)

    if hmax.shape[2] > heat.shape[2]:
        hmax = hmax[:, :, : heat.shape[2], : heat.shape[3]]

    keep = (hmax == heat).float()

    return heat * keep


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


if __name__ == "__main__":
    Model()