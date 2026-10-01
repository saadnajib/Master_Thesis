"""
Knowledge-distillation loss for Multi-HMR backbone compression.

PLACEMENT (per supervisor's instruction): the loss is applied to the backbone's
patch tokens - i.e. immediately after patch decoding and BEFORE the projection
heads (detection MLP, HPH, pose/shape/dist MLPs). Concretely it compares
    teacher: z_t = teacher_backbone(x)   # [B, N, C_t]
    student: z_s = student_backbone(x)   # [B, N, C_s]
with no head involvement, so the student learns to reproduce the teacher's
REPRESENTATION rather than only its final predictions.

WHY KL AND NOT L2
-----------------
KL divergence compares probability DISTRIBUTIONS, so the raw feature vectors
must be turned into distributions first. Following Hinton et al. (2015) this is
a temperature-scaled softmax; the gradient is then multiplied by T^2 so the
loss magnitude stays comparable as T changes (without that correction, raising
T silently shrinks the distillation gradient).

The softmax is taken over the CHANNEL dimension by default: each patch token
becomes a distribution over feature channels, and the student is asked to match
the teacher's per-patch channel pattern. Softmaxing over tokens instead
(--kd_softmax_dim token) asks it to match the spatial attention pattern - a
different and usually weaker objective. Channel is the standard choice.

DIMENSION MISMATCH
------------------
ViT-L has C=1024 channels, ViT-B 768, ViT-S 384. KL needs both distributions
over the SAME support, so when the student is narrower a learnable linear
projection maps student channels up to teacher width. This projector is part of
the student's trainable parameters and is DISCARDED at inference - it exists
only to make the loss computable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceLoss(nn.Module):
    """KL(teacher || student) on temperature-softened backbone patch tokens.

    Args:
        student_dim: student backbone channel width (384 ViT-S, 768 ViT-B)
        teacher_dim: teacher backbone channel width (1024 for ViT-L)
        temperature: softmax temperature. Higher = softer targets, more of the
            teacher's "dark knowledge" (relative magnitudes of non-peak
            channels) is transferred. 1.0 approximates a hard match; very high
            values flatten everything toward uniform. 4.0 is the common default.
        softmax_dim: 'channel' (default) or 'token' - see module docstring.
    """

    def __init__(self, student_dim, teacher_dim, temperature=4.0,
                 softmax_dim='channel'):
        super().__init__()
        assert softmax_dim in ('channel', 'token')
        self.T = float(temperature)
        self.softmax_dim = softmax_dim

        # Only needed when the widths differ. Identity keeps the graph clean
        # (and adds no parameters) when student and teacher are the same size.
        if student_dim != teacher_dim:
            self.proj = nn.Linear(student_dim, teacher_dim)
            self.needs_proj = True
        else:
            self.proj = nn.Identity()
            self.needs_proj = False

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: [B, N, C_s] patch tokens from the student backbone
            teacher_feat: [B, N, C_t] patch tokens from the frozen teacher
        Returns:
            scalar loss
        """
        if student_feat.shape[:2] != teacher_feat.shape[:2]:
            raise ValueError(
                f"token grids differ: student {tuple(student_feat.shape)} vs "
                f"teacher {tuple(teacher_feat.shape)}. Both backbones must see "
                f"the same --img_size and use the same patch size (all DINOv2 "
                f"variants are /14, so this usually means mismatched img_size)."
            )

        s = self.proj(student_feat)                 # [B, N, C_t]
        t = teacher_feat.detach()                   # teacher is frozen

        dim = -1 if self.softmax_dim == 'channel' else 1

        # KLDivLoss expects log-probabilities for the input and probabilities
        # for the target, and computes KL(target || input).
        s_log = F.log_softmax(s / self.T, dim=dim)
        t_prob = F.softmax(t / self.T, dim=dim)

        kl = F.kl_div(s_log, t_prob, reduction='batchmean')

        # T^2 keeps gradient magnitude stable across temperatures (Hinton et al.)
        return kl * (self.T ** 2)


def build_teacher(ckpt_path, device, verbose=True):
    """Load a frozen teacher Model from one of our own checkpoints.

    Returns (teacher_model, teacher_dim). The teacher is put in eval mode with
    requires_grad=False on every parameter - it must never be updated, and its
    BatchNorm/dropout statistics must not drift.
    """
    from model import Model

    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state_dict' not in ckpt:
        raise ValueError(f"{ckpt_path} has no 'model_state_dict'")

    kwargs = dict(vars(ckpt['args'])) if 'args' in ckpt else {}
    kwargs.setdefault('simple_depth_encoding', 1)
    kwargs['pretrained_backbone'] = 0      # weights come from the checkpoint
    kwargs.pop('pretrained', None)

    teacher = Model(**kwargs).to(device)

    sd = ckpt['model_state_dict']
    msd = teacher.state_dict()
    matched = [k for k, v in sd.items()
               if k in msd and hasattr(v, 'shape') and msd[k].shape == v.shape]
    teacher.load_state_dict(sd, strict=False)

    pct = 100.0 * len(matched) / max(len(msd), 1)
    if verbose:
        print(f"[distill] teacher loaded from {ckpt_path}: "
              f"{len(matched)}/{len(msd)} tensors ({pct:.1f}%)", flush=True)
    # Same silent-failure guard used for --pretrained: a mismatched teacher
    # would emit garbage features and quietly poison the student.
    if pct < 80.0:
        raise RuntimeError(
            f"Only {pct:.1f}% of the teacher was initialised from {ckpt_path}. "
            f"Refusing to distill from a mostly-random teacher.")

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    teacher_dim = teacher.backbone.embed_dim
    if verbose:
        n_par = sum(p.numel() for p in teacher.parameters())
        print(f"[distill] teacher frozen: {n_par:,} params, "
              f"embed_dim={teacher_dim}", flush=True)
    return teacher, teacher_dim
