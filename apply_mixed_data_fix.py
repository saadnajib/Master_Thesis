"""
apply_mixed_data_fix.py -- train on AnnyOne + converted 3DPW, evaluate on 3DPW.

Adds to train.py (all default OFF, old runs unchanged):
  --mix_3dpw_labels DIR   root of the labels written by build_threedpw_anny_labels.py
                          (the one with index_train.txt). When set, the 3DPW
                          '--mix_3dpw_split' images are concatenated to the
                          AnnyOne training set and a WeightedRandomSampler
                          draws every sample so that on average a fraction
                          --mix_3dpw_ratio of each batch is real 3DPW.
  --mix_3dpw_ratio R      default 0.3
  --mix_3dpw_split S      default train     (NEVER test)
  --val_3dpw_split S      also evaluate every --eval_freq epochs on this 3DPW
                          split of the same labels (default validation; '' = off).
                          Uses the existing evaluate(): PVE / PA-PVE / MPJPE on
                          Anny vertices against the converted ground truth, so
                          the ~13-20 mm conversion error is a floor on it.
  --val_3dpw_n N          evaluate on N evenly spaced images (0 = all)

Needs datasets/threedpw_anny.py (copy it into datasets/).

USAGE (in /netscratch/najib/multi-hmr):
    cp train.py train.py.bak_before_mixed
    cp threedpw_anny.py datasets/threedpw_anny.py
    python apply_mixed_data_fix.py
"""
import sys

TRAIN = 'train.py'


def patch(path, edits, marker):
    s = open(path).read()
    if marker in s:
        print(f"{path}: already patched (mixed data) - nothing to do")
        return
    for anchor, new, what in edits:
        n = s.count(anchor)
        if n != 1:
            print(f"ERROR: anchor for {what} found {n} times in {path} (need exactly 1):\n{anchor!r}")
            print("Send me the file and I will regenerate the patch."); sys.exit(1)
        s = s.replace(anchor, new)
    open(path, 'w').write(s)
    print(f"patched {path}: " + ", ".join(e[2] for e in edits))


# 1. flags
t_flag = "    parser.add_argument('--freeze_heads_epochs', type=int, default=0,"
t_flag_new = """    parser.add_argument('--mix_3dpw_labels', type=str, default='',
                        help='root of the converted 3DPW Anny labels (has index_train.txt). '
                             'When set, real 3DPW images are mixed into AnnyOne training.')
    parser.add_argument('--mix_3dpw_ratio', type=float, default=0.3,
                        help='average fraction of real 3DPW samples per batch')
    parser.add_argument('--mix_3dpw_split', type=str, default='train',
                        help='3DPW split used for TRAINING (never test)')
    parser.add_argument('--val_3dpw_split', type=str, default='validation',
                        help="3DPW split of the same labels to evaluate on every eval ('' = off)")
    parser.add_argument('--val_3dpw_n', type=int, default=0,
                        help='evaluate on N evenly spaced 3DPW images (0 = all)')
""" + t_flag

# 2. mixing, right after the train subset / holdout split and before the collate def
t_mix = "            # Define it RIGHT HERE, after train_dataset exists\n"
t_mix_new = """            # --- mixed synthetic + real training (apply_mixed_data_fix.py) ----
            sampler = None
            if getattr(args, 'mix_3dpw_labels', ''):
                from torch.utils.data import ConcatDataset, WeightedRandomSampler
                from datasets.threedpw_anny import ThreeDPWAnny
                if args.mix_3dpw_split == 'test':
                    raise ValueError("--mix_3dpw_split test: never train on the 3DPW test split")
                real_dataset = ThreeDPWAnny(args.mix_3dpw_labels, split=args.mix_3dpw_split,
                                            img_size=args.img_size)
                n_syn, n_real = len(train_dataset), len(real_dataset)
                if n_real == 0:
                    raise RuntimeError(f"no 3DPW labels found under {args.mix_3dpw_labels}/{args.mix_3dpw_split}")
                r = float(args.mix_3dpw_ratio)
                weights = torch.cat([torch.full((n_syn,), (1.0 - r) / n_syn),
                                     torch.full((n_real,), r / n_real)])
                train_dataset = ConcatDataset([train_dataset, real_dataset])
                sampler = WeightedRandomSampler(weights, num_samples=n_syn + n_real, replacement=True)
                print(f"[mixed] training on {n_syn} AnnyOne + {n_real} 3DPW-{args.mix_3dpw_split} images; "
                      f"on average {100 * r:.0f}% of every batch is real 3DPW", flush=True)

            # Define it RIGHT HERE, after train_dataset exists
"""

# 3. the loader uses the sampler when there is one
t_dl = """            train_data = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=bulletproof_collate,
"""
t_dl_new = """            train_data = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=(sampler is None),
                sampler=sampler,
                collate_fn=bulletproof_collate,
"""

# 4. 3DPW-Anny evaluation loader, after the AnnyOne holdout block
t_val = """                else:
                    print("WARNING: --val_anny_n requested but no samples left after the train subset", flush=True)
"""
t_val_new = t_val + """
            # --- evaluation on converted 3DPW (apply_mixed_data_fix.py) ---------
            if getattr(args, 'mix_3dpw_labels', '') and getattr(args, 'val_3dpw_split', ''):
                from datasets.threedpw_anny import ThreeDPWAnny
                val_3dpw = ThreeDPWAnny(args.mix_3dpw_labels, split=args.val_3dpw_split,
                                        img_size=args.img_size, n=args.val_3dpw_n, training=False)
                if len(val_3dpw) > 0:
                    l_val_data.append(DataLoader(val_3dpw, batch_size=1, shuffle=False, num_workers=0,
                                                 drop_last=False, collate_fn=bulletproof_collate))
                    print(f"3DPW-Anny val ({args.val_3dpw_split}): {len(val_3dpw)} images", flush=True)
"""

if __name__ == '__main__':
    patch(TRAIN, [(t_flag, t_flag_new, '+--mix_3dpw_* flags'),
                  (t_mix, t_mix_new, 'ConcatDataset + weighted sampler'),
                  (t_dl, t_dl_new, 'loader uses sampler'),
                  (t_val, t_val_new, '3DPW-Anny eval loader')],
          marker='mix_3dpw_labels')
