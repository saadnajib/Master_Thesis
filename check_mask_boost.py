"""
Check whether the helper-joint MASK and the per-joint loss BOOST conflict.

THE QUESTION THIS ANSWERS
-------------------------
Two of our fixes may be cancelling each other out:

  * --mask_helper_joints_train 1 forces 88-89 "helper" bones to identity in
    BOTH the prediction and the GT, so their rotmat loss is exactly ZERO.
  * --boost_neck_weight 4 / --boost_hand_weight 4 multiply the loss on
    neck/head and finger/thumb/wrist bones by 4.

If a boosted bone is also a masked bone, the boost multiplies zero by four,
which is still zero. That bone can never learn - which would explain why the
v4 run improved general pose but left the neck unchanged.

This script prints, for every neck/head/hand bone, whether it is MASKED
(loss dead) or ACTIVE (loss live and boosted).

USAGE (login node is fine, no GPU needed):
    python check_mask_boost.py
"""

import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

# The useful_rotmat mask, copied verbatim from multi_hmr_anny/multi_hmr.py.
# 1 = bone keeps its predicted rotation (loss is live)
# 0 = bone is forced to identity (loss is exactly zero)
USEFUL_ROTMAT = [
    1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0.,
]


def is_neck(name):
    lo = name.lower()
    return ('neck' in lo) or (lo == 'head')


def is_hand(name):
    lo = name.lower()
    return ('finger' in lo) or ('thumb' in lo) or ('wrist' in lo)


def main():
    try:
        import anny
    except ImportError as e:
        print(f"ERROR: could not import anny ({e}).")
        print("Run this inside the multihmr conda env:  conda activate multihmr")
        sys.exit(1)

    print("Loading Anny body model ...")
    m = anny.create_fullbody_model(remove_unattached_vertices=False,
                                   all_phenotypes=True)
    names = list(m.bone_labels)
    n = len(names)
    print(f"body model has {n} bones; mask vector has {len(USEFUL_ROTMAT)} entries")
    if len(USEFUL_ROTMAT) != n:
        print(f"WARNING: length mismatch! Mask is {len(USEFUL_ROTMAT)}, bones are {n}.")
        print("         This alone is a bug worth fixing - the mask may be")
        print("         mis-aligned with the actual bone ordering.")

    def masked(i):
        return i < len(USEFUL_ROTMAT) and USEFUL_ROTMAT[i] == 0.0

    print("\n" + "=" * 72)
    print("NECK / HEAD BONES  (targeted by --boost_neck_weight)")
    print("=" * 72)
    neck_rows = [(i, nm) for i, nm in enumerate(names) if is_neck(nm)]
    n_neck_masked = 0
    for i, nm in neck_rows:
        if masked(i):
            n_neck_masked += 1
            status = "MASKED  -> loss is ZERO, boost does NOTHING"
        else:
            status = "active  -> loss is live, boost applies"
        print(f"  [{i:3d}] {nm:<28s} {status}")
    if not neck_rows:
        print("  (none matched 'neck' or 'head' - check the naming convention!)")

    print("\n" + "=" * 72)
    print("HAND BONES  (targeted by --boost_hand_weight)")
    print("=" * 72)
    hand_rows = [(i, nm) for i, nm in enumerate(names) if is_hand(nm)]
    n_hand_masked = 0
    for i, nm in hand_rows[:40]:
        if masked(i):
            n_hand_masked += 1
            status = "MASKED  -> loss is ZERO, boost does NOTHING"
        else:
            status = "active  -> loss is live, boost applies"
        print(f"  [{i:3d}] {nm:<28s} {status}")
    n_hand_masked = sum(1 for i, nm in hand_rows if masked(i))
    if len(hand_rows) > 40:
        print(f"  ... and {len(hand_rows) - 40} more")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  neck/head bones: {len(neck_rows):3d} total, "
          f"{n_neck_masked} masked (loss dead), "
          f"{len(neck_rows) - n_neck_masked} active")
    print(f"  hand bones:      {len(hand_rows):3d} total, "
          f"{n_hand_masked} masked (loss dead), "
          f"{len(hand_rows) - n_hand_masked} active")
    print()
    if n_neck_masked > 0:
        print("  >>> CONFLICT CONFIRMED for the NECK.")
        print("      Some neck bones are masked to identity, so their loss is")
        print("      exactly zero and --boost_neck_weight had no effect on them.")
        print("      FIX: exclude neck bones from the helper mask so they can")
        print("      learn, then continue training.")
    else:
        print("  >>> No neck conflict: every neck/head bone is active in the loss.")
        print("      The boost DID apply to them, so the remaining neck error is")
        print("      not explained by masking. Look to the data / domain gap")
        print("      instead (see the AnnyOne in-domain demo).")
    print()
    if n_hand_masked > 0:
        print("  >>> CONFLICT for the HANDS: some finger/thumb/wrist bones are")
        print("      masked, so --boost_hand_weight was partly wasted too.")
    else:
        print("  >>> No hand conflict: hand bones are active in the loss.")

    # Extra context: how many bones are dead in total.
    total_masked = sum(1 for i in range(min(n, len(USEFUL_ROTMAT))) if masked(i))
    print(f"\n  (for reference: {total_masked} of {n} bones are masked overall)")


if __name__ == '__main__':
    main()
