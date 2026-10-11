"""
apply_speed_fix.py -- make distillation iterations faster (train.py + model.py).

WHERE THE TIME GOES (job 3455531: 1.24 s/it, batch 8, 672 px, --amp 0)
------------------------------------------------------------------------
The frozen ViT-L teacher processes 2,304 tokens x 24 layers = ~1.9 TFLOP per
image, ~15 TFLOP per batch of 8, every iteration. train.py never enables TF32,
so with --amp 0 those matmuls run in plain fp32 (~19 TFLOPS on an A100): that
is ~1.2 s on its own. The ViT-S student (14x fewer FLOPs) and the losses are
small next to it.

WHAT THIS PATCH ADDS
--------------------
  --tf32 1 (default)      torch.backends.cuda.matmul.allow_tf32 = True (+cudnn).
                          fp32 range, 10-bit mantissa inputs, fp32 accumulation;
                          PyTorch 2.0 leaves it OFF for matmul by default. ~2-3x
                          on Ampere/Ada (A100, L40S, RTX A6000). --tf32 0 = old.
  --teacher_bf16 1        the TEACHER's backbone runs under bfloat16 autocast.
                          It is no_grad, so no gradient scaling is involved, and
                          bf16 keeps fp32's exponent range (no overflow). The
                          tokens are cast back to fp32 BEFORE the heads and the
                          KD loss, so the rotmat/shape targets for output
                          distillation keep full precision. Default 0 so earlier
                          runs reproduce exactly; pass 1 in new job scripts.
  DataLoader              pin_memory=True, persistent_workers=True on the
                          training loader (workers no longer restart per epoch).

NOT CHANGED: the student's own --amp. In this train.py it is fp16 autocast
with no GradScaler, which is the classic recipe for NaN gradients. Leave it 0.

USAGE (in /netscratch/najib/multi-hmr):
    cp train.py train.py.bak_before_speed ; cp model.py model.py.bak_before_speed
    python apply_speed_fix.py
"""
import sys

MODEL, TRAIN = 'model.py', 'train.py'


def patch(path, edits, marker):
    s = open(path).read()
    if marker in s:
        print(f"{path}: already patched (speed) - nothing to do")
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
m_fwd = "        z = self.backbone(x)\n"
m_fwd_new = """        # --- optional reduced-precision BACKBONE (apply_speed_fix.py) --------
        # Set model.backbone_autocast_dtype = torch.bfloat16 on a frozen teacher:
        # the backbone runs in bf16, its tokens are cast back to fp32 before the
        # heads so every downstream quantity keeps full precision. None = fp32.
        _bb_dtype = getattr(self, 'backbone_autocast_dtype', None)
        if _bb_dtype is not None and x.is_cuda:
            with torch.autocast('cuda', dtype=_bb_dtype):
                z = self.backbone(x)
            z = z.float()
        else:
            z = self.backbone(x)
"""

# ============================================================ train.py
t_flag = "    parser.add_argument('--freeze_heads_epochs', type=int, default=0,"
t_flag_new = """    parser.add_argument('--tf32', type=int, default=1, choices=[0, 1],
                        help='allow TensorFloat-32 matmuls/convs on Ampere+ GPUs (fp32 range, '
                             '~2-3x faster). 0 reproduces the old plain-fp32 behaviour.')
    parser.add_argument('--teacher_bf16', type=int, default=0, choices=[0, 1],
                        help='run the frozen TEACHER backbone under bfloat16 autocast (no_grad, '
                             'no overflow risk); tokens are cast back to fp32 before its heads '
                             'and the KD loss.')
""" + t_flag

t_dev = "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
t_dev_new = """    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # --- matmul precision (apply_speed_fix.py) -------------------------------
    if int(getattr(args, 'tf32', 1)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("[speed] TF32 matmul/conv ENABLED (fp32 range, 10-bit mantissa inputs)", flush=True)
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        print("[speed] TF32 disabled (plain fp32 matmuls)", flush=True)
"""

t_teacher = "        teacher, teacher_dim = build_teacher(args.distill_teacher_ckpt, device)\n"
t_teacher_new = """        teacher, teacher_dim = build_teacher(args.distill_teacher_ckpt, device)
        if int(getattr(args, 'teacher_bf16', 0)):
            teacher.backbone_autocast_dtype = torch.bfloat16
            print("[speed] teacher backbone runs in bfloat16 (tokens cast to fp32 for heads + KD)", flush=True)
"""

t_dl1 = """                collate_fn=bulletproof_collate,
                num_workers=args.num_workers,
                drop_last=True,
            )
"""
t_dl1_new = """                collate_fn=bulletproof_collate,
                num_workers=args.num_workers,
                drop_last=True,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
            )
"""
t_dl2 = """                num_workers=args.num_workers,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn,
            )
"""
t_dl2_new = """                num_workers=args.num_workers,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
            )
"""

if __name__ == '__main__':
    patch(MODEL, [(m_fwd, m_fwd_new, 'backbone autocast hook')], marker='backbone_autocast_dtype')
    patch(TRAIN, [(t_flag, t_flag_new, '+--tf32/--teacher_bf16 flags'),
                  (t_dev, t_dev_new, 'TF32 switch'),
                  (t_teacher, t_teacher_new, 'teacher bf16 backbone'),
                  (t_dl1, t_dl1_new, 'AnnyOne loader pin/persistent'),
                  (t_dl2, t_dl2_new, 'BEDLAM loader pin/persistent')],
          marker='--teacher_bf16')
