"""
apply_student_fix.py -- patch train.py for the distillation student.

WHY (measured, not guessed)
---------------------------
Student (ViT-S) holdout PVE = 193.4 mm vs teacher (ViT-L) 73.6 mm: 2.6x worse.
The Multi-HMR paper's own ViT-S vs ViT-B gap is only 80 vs 73 mm, so this is
not a capacity limit. Cause: the KD loss supervised ONLY the backbone tokens.
The HPH / pose / shape heads - the parts that turn features into a body - were
randomly initialised and learned from scratch by a small model on synthetic
data, while the teacher's heads were trained through five stages. Result: the
student detects and places people (backbone job) but produces wrong poses
(head job). PVE was still crawling down (198 -> 193 over 80 epochs), i.e. a
bad start, not convergence.

WHAT THIS PATCH ADDS (two flags, both default OFF so old runs are unchanged)
---------------------------------------------------------------------------
  --init_heads_from_teacher 1
      Copy every teacher tensor whose name and shape match the student's,
      EXCEPT the backbone. Teacher and student are the same Model class, so
      the HPH blocks and the pose/shape/depth MLPs transfer exactly; only the
      few input projections that touch embed_dim (1024 vs 384) are left random
      and are listed in the log. The student starts with trained heads.

  --lambda_kd_out <w>   (e.g. 1.0)
      Output-level distillation: L1 between the student's and the teacher's
      predicted rotmat / shape / depth on the same image, weighted like the
      task terms (alpha_rotmat, alpha_shape, alpha_dist). Both models run on
      the same GT query locations in training mode, so person i of the student
      corresponds to person i of the teacher and no matching step is needed.
      This gives the heads direct supervision to behave like the teacher, on
      top of the feature-level KL that supervises the backbone.

USAGE (on the cluster, in /netscratch/najib/multi-hmr):
    cp train.py train.py.bak_before_student_fix
    python apply_student_fix.py train.py
It refuses to run twice and refuses if any anchor is missing.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'train.py'
s = open(path).read()

if 'init_heads_from_teacher' in s:
    print("already patched - nothing to do"); sys.exit(0)

def must(anchor, what):
    if anchor not in s:
        print(f"ERROR: could not find anchor for {what}:\n{anchor!r}")
        print("train.py differs from the version this patch expects. "
              "Send me the file and I will regenerate the patch.")
        sys.exit(1)
    if s.count(anchor) != 1:
        print(f"ERROR: anchor for {what} is not unique ({s.count(anchor)} hits)")
        sys.exit(1)

# ---------------------------------------------------------------- 1. CLI flags
a_flag = "    parser.add_argument('--kd_softmax_dim', type=str, default='channel',"
must(a_flag, "kd_softmax_dim flag")
flags = """    parser.add_argument('--init_heads_from_teacher', type=int, default=0, choices=[0, 1],
                        help='copy every non-backbone tensor whose name AND shape match from the '
                             'teacher into the student before training (HPH, pose/shape/depth '
                             'MLPs). Layers touching embed_dim stay random and are listed.')
    parser.add_argument('--lambda_kd_out', type=float, default=0.0,
                        help='weight of OUTPUT-level distillation: L1 between student and teacher '
                             'predicted rotmat/shape/depth on the same image. 0 = off.')
"""
s = s.replace(a_flag, flags + a_flag)

# ------------------------------------------------- 2. head init in main()
a_teacher = "        teacher, teacher_dim = build_teacher(args.distill_teacher_ckpt, device)\n"
must(a_teacher, "teacher construction")
head_init = a_teacher + """
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
"""
s = s.replace(a_teacher, head_init)

# ----------------------------------------- 3. output distillation in the loop
a_kd = """                    loss = loss + self.args.lambda_kd * kd
                    dict_loss['kd'] = kd
                    dict_loss['total'] = loss
"""
must(a_kd, "feature-KD block in train_n_iters")
out_kd = """                    loss = loss + self.args.lambda_kd * kd
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
"""
s = s.replace(a_kd, out_kd)

open(path, 'w').write(s)
print(f"patched {path}: +2 flags, head-init block, output-KD block")
