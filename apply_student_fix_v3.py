"""
apply_student_fix_v3.py -- feed the heads PROJECTED tokens (--head_dim).

Run AFTER apply_student_fix.py and apply_student_fix_v2.py. Patches BOTH
model.py and train.py. Refuses to run twice and refuses on a missing anchor.

WHY (measured in v3 / v4 / v5, jobs 3453670 / 3453700 / 3454122)
------------------------------------------------------------------
Every student with teacher-initialised heads was WORSE than the random-heads
student at the same epoch (v3 at epoch 45: PVE 274 vs v1's 228 at epoch 0),
and PA-PVE stayed at 155-199 in all three against 128 for random heads. The
5-epoch head freeze (v5) protected the transferred weights - rotmat loss fell
to 90 where v4's sat at 300 - but PVE at the unfreeze eval was 315 and PA-PVE
did not move (192 -> 193). So the heads are intact and still produce bad poses.

The 13 tensors that could not transfer are the head's entire INPUT side:
HPH to_token_embedding, the cross-attention K/V projections, mlp_classif and
mlp_offset. They are linear maps from the student's 384-dim tokens that must
reproduce what the teacher's 1024-dim tokens produced through the teacher's
own projections. Meanwhile the KD projector - the one layer that IS trained
(by the feature KL) to map 384-dim student tokens onto the teacher's token
distribution - lives inside the loss and is thrown away; the heads never see
its output.

FIX: --head_dim 1024
    Model gets an nn.Linear(backbone_dim, head_dim) right after the backbone.
    Every head (detection, offset, HPH) is built at head_dim, so with
    head_dim = teacher width ALL 67 head tensors transfer exactly
    (--init_heads_from_teacher 1 -> "67/67"). The projected tokens are also
    what out['backbone_feat'] exposes, so the feature KL compares them with
    the teacher's tokens directly (same width, no loss-side projector) and
    trains the projector towards the input distribution the heads were
    trained on. Inference cost: one 384->1024 linear per token (~0.4M params).
    --head_dim 0 (default) keeps the old behaviour byte-for-byte.

    Pair with --freeze_heads_epochs N: during the warm-up only the backbone and
    the projector train, driven by feature KD (match the teacher's tokens) and
    output KD (match the teacher's predictions THROUGH the teacher's own heads).

USAGE (on the cluster, in /netscratch/najib/multi-hmr):
    cp model.py model.py.bak_before_v3
    cp train.py train.py.bak_before_v3
    python apply_student_fix_v3.py
"""
import sys

MODEL, TRAIN = 'model.py', 'train.py'


def patch(path, edits, marker):
    s = open(path).read()
    if marker in s:
        print(f"{path}: already patched (v3) - nothing to do")
        return
    for anchor, new, what in edits:
        n = s.count(anchor)
        if n != 1:
            print(f"ERROR: anchor for {what} found {n} times in {path} (need exactly 1):\n{anchor!r}")
            print("Send me the file and I will regenerate the patch."); sys.exit(1)
        s = s.replace(anchor, new)
    open(path, 'w').write(s)
    print(f"patched {path}: " + ", ".join(e[2] for e in edits))


# ============================================================ model.py
m_sig = "        num_betas=11,\n        *args,\n        **kwargs,\n    ):\n        super().__init__()\n\n        self.img_size = img_size\n"
m_sig_new = "        num_betas=11,\n        head_dim=None,\n        *args,\n        **kwargs,\n    ):\n        super().__init__()\n\n        self.img_size = img_size\n"

m_dim = "        self.embed_dim = self.backbone.embed_dim\n        self.patch_size = self.backbone.patch_size\n"
m_dim_new = """        self.embed_dim = self.backbone.embed_dim
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
"""

m_fwd = "        z = self.backbone(x)\n        B, N, C = z.size()\n"
m_fwd_new = """        z = self.backbone(x)
        if self.feat_proj is not None:
            out['backbone_feat_raw'] = z          # [B, N, backbone_dim], for reference
            z = self.feat_proj(z)                 # [B, N, head_dim] - what the heads see
        B, N, C = z.size()
"""

# ============================================================ train.py
t_flag = "    parser.add_argument('--freeze_heads_epochs', type=int, default=0,"
t_flag_new = """    parser.add_argument('--head_dim', type=int, default=0,
                        help='student only: width the heads are built at, with a linear projector '
                             'after the backbone. Set to the TEACHER width (1024 for ViT-L) so every '
                             'head tensor transfers with --init_heads_from_teacher. 0 = off.')
""" + t_flag

t_dim = "        student_dim = model.backbone.embed_dim\n"
t_dim_new = """        # Width of the tokens the heads (and the KD tap) see. With --head_dim this
        # is the projector output, equal to the teacher width, so the KD loss
        # needs no projector of its own.
        student_dim = model.embed_dim
        if getattr(model, 'feat_proj', None) is not None:
            print(f"[distill] projector-fed heads: backbone {model.backbone_dim} -> heads {model.embed_dim} "
                  f"(feat_proj {sum(p.numel() for p in model.feat_proj.parameters()):,} params); "
                  f"feature KD is computed on the PROJECTED tokens", flush=True)
"""

t_opt = "        optimizer.add_param_group({'params': kd_loss.parameters()})\n"
t_opt_new = """        _kd_params = list(kd_loss.parameters())
        if _kd_params:
            optimizer.add_param_group({'params': _kd_params})
"""

t_warn = """        if student_dim == teacher_dim:
            print("[distill] WARNING: student and teacher have the SAME width - "
                  "is --backbone actually set to a smaller ViT?", flush=True)
"""
t_warn_new = """        if student_dim == teacher_dim and getattr(model, 'feat_proj', None) is None:
            print("[distill] WARNING: student and teacher have the SAME width - "
                  "is --backbone actually set to a smaller ViT?", flush=True)
"""

if __name__ == '__main__':
    s_train = open(TRAIN).read()
    if 'freeze_heads_epochs' not in s_train or 'init_heads_from_teacher' not in s_train:
        print("ERROR: run apply_student_fix.py and apply_student_fix_v2.py first"); sys.exit(1)
    patch(MODEL, [(m_sig, m_sig_new, '+head_dim arg'),
                  (m_dim, m_dim_new, 'feat_proj construction'),
                  (m_fwd, m_fwd_new, 'projection in forward')],
          marker='feat_proj')
    patch(TRAIN, [(t_flag, t_flag_new, '+--head_dim flag'),
                  (t_dim, t_dim_new, 'KD width = head width'),
                  (t_opt, t_opt_new, 'skip empty KD param group'),
                  (t_warn, t_warn_new, 'same-width warning')],
          marker='--head_dim')
