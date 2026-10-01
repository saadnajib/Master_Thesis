"""
apply_student_fix_v2.py -- add --freeze_heads_epochs to train.py.

Run AFTER apply_student_fix.py (it requires the head-init block that one adds).

WHY (measured in v3 / v4, jobs 3453670 / 3453700)
--------------------------------------------------
Teacher-initialised heads made the FIRST eval WORSE than random heads:
    random heads  (v1/v2): first-eval PVE 228
    teacher heads (v3):    376.8, still 370.5 at epoch 5
    teacher heads (v4):    298.2
The log lists the 13 tensors that could not transfer, and they are exactly
the head's ENTRY POINTS:
    x_attention_head.transformer.to_token_embedding.weight   (input projection)
    x_attention_head.cross_queries_{x,y}, cross_values_{x,y} (cross-attn K/V)
    x_attention_head...layers.{0,1}.1.fn.to_kv.weight        (cross-attn K/V)
    mlp_classif.*, mlp_offset.*                              (detection, offset)
So the well-trained HPH blocks and pose/shape MLPs receive image features
through RANDOM projections. A trained head fed garbage produces confident
wrong poses - worse than a random head producing neutral ones - and the
gradients from those wrong outputs then start damaging the transferred
weights before the projections have learned to feed them.

FIX: --freeze_heads_epochs N
    For the first N epochs, every tensor that was copied from the teacher has
    requires_grad=False. Only the 13 random projections, the backbone and the
    KD projector train, with the output-distillation term giving them a strong
    target (match the teacher's predictions). At epoch N everything unfreezes.
    Pair with --start_2d_epoch N so reprojection gradients also wait until
    the projections are sensible.

USAGE
    python apply_student_fix_v2.py train.py
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'train.py'
s = open(path).read()

if 'freeze_heads_epochs' in s:
    print("already patched (v2) - nothing to do"); sys.exit(0)
if 'init_heads_from_teacher' not in s:
    print("ERROR: run apply_student_fix.py first (head-init block missing)"); sys.exit(1)

def must(anchor, what):
    n = s.count(anchor)
    if n != 1:
        print(f"ERROR: anchor for {what} found {n} times (need exactly 1):\n{anchor!r}")
        print("Send me train.py and I will regenerate the patch."); sys.exit(1)

# 1. flag, right after --lambda_kd_out
a1 = "    parser.add_argument('--lambda_kd_out', type=float, default=0.0,"
must(a1, "lambda_kd_out flag")
s = s.replace(a1, """    parser.add_argument('--freeze_heads_epochs', type=int, default=0,
                        help='with --init_heads_from_teacher: keep every tensor copied from the '
                             'teacher FROZEN for this many epochs so the random embed_dim '
                             'projections learn to feed them first. 0 = off.')
""" + a1)

# 2. remember which keys came from the teacher
a2 = "            model.load_state_dict(sd_s)\n            n_head = sum(1 for k in sd_s if not k.startswith('backbone.'))\n"
must(a2, "head-init load_state_dict")
s = s.replace(a2, """            model.load_state_dict(sd_s)
            # remembered for --freeze_heads_epochs (see train_n_iters)
            model._teacher_head_keys = set(copied)
            n_head = sum(1 for k in sd_s if not k.startswith('backbone.'))
""")

# 3. freeze/unfreeze at the start of every epoch
a3 = "    def train_n_iters(self, data):\n"
must(a3, "train_n_iters def")
s = s.replace(a3, a3 + """        # --- head warm-up freeze (see --freeze_heads_epochs) -----------------
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
""")

open(path, 'w').write(s)
print(f"patched {path}: +--freeze_heads_epochs, head-key bookkeeping, per-epoch freeze")
