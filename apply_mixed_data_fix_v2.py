"""
apply_mixed_data_fix_v2.py -- 3DPW-Anny evaluation in --eval_only mode.

apply_mixed_data_fix.py wired the converted-3DPW evaluation loader into the
TRAINING branch of train.py only. With --eval_only 1 --test_anny_n N the code
takes the test-split branch instead, so run_eval_3dpw.sh printed the Anny-One
block and silently skipped the 3DPW one (jobs 3502346 / 3502348). This adds the
same loader right after the Anny-One test subset in that branch.

Run AFTER apply_mixed_data_fix.py.   python apply_mixed_data_fix_v2.py
"""
import sys

TRAIN = 'train.py'
s = open(TRAIN).read()
if 'mix_3dpw_labels' not in s:
    print("ERROR: run apply_mixed_data_fix.py first"); sys.exit(1)
if 'eval-only 3DPW-Anny' in s:
    print("already patched (v2) - nothing to do"); sys.exit(0)

anchor = """            l_val_data.append(DataLoader(test_subset, batch_size=1, shuffle=False,
                                         num_workers=args.num_workers, drop_last=False,
                                         collate_fn=_test_collate))
"""
n = s.count(anchor)
if n != 1:
    print(f"ERROR: anchor found {n} times (need 1). Send me train.py."); sys.exit(1)
new = anchor + """
            # --- eval-only 3DPW-Anny test loader (apply_mixed_data_fix_v2.py) --
            if getattr(args, 'mix_3dpw_labels', '') and getattr(args, 'val_3dpw_split', ''):
                from datasets.threedpw_anny import ThreeDPWAnny
                eval_3dpw = ThreeDPWAnny(args.mix_3dpw_labels, split=args.val_3dpw_split,
                                         img_size=args.img_size, n=args.val_3dpw_n, training=False)
                if len(eval_3dpw) > 0:
                    l_val_data.append(DataLoader(eval_3dpw, batch_size=1, shuffle=False,
                                                 num_workers=args.num_workers, drop_last=False,
                                                 collate_fn=_test_collate))
                    print(f"3DPW-Anny eval ({args.val_3dpw_split}): {len(eval_3dpw)} images", flush=True)
"""
s = s.replace(anchor, new)
open(TRAIN, 'w').write(s)
print(f"patched {TRAIN}: eval-only 3DPW-Anny loader")
