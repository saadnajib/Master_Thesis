import sys, os, datetime, warnings
warnings.filterwarnings("ignore")

sys.path.append('/netscratch/najib/multi-hmr/')

# --- REDIRECT CACHES OFF THE OVER-QUOTA HOME DIR ---
os.environ.setdefault('XDG_CACHE_HOME', '/netscratch/najib/.cache')
os.environ.setdefault('HF_HOME', '/netscratch/najib/hf_cache')
os.environ.setdefault('TORCH_HOME', '/netscratch/najib/torch_cache')
os.makedirs(os.environ['XDG_CACHE_HOME'], exist_ok=True)

# Best-effort: make anny's hardcoded ~/.cache/anny point to netscratch, so
# Part 3 does not die on the home-dir quota. Safe if it already exists.
try:
    home_cache = os.path.expanduser('~/.cache/anny')
    target = '/netscratch/najib/anny_cache'
    os.makedirs(target, exist_ok=True)
    if not os.path.exists(home_cache):
        os.makedirs(os.path.dirname(home_cache), exist_ok=True)
        os.symlink(target, home_cache)
except Exception:
    pass  # if this fails, Part 3 may still hit the quota; Parts 1/2 are enough
# ---------------------------------------------------

# --- MOCK GRAPHICS LIBRARIES FOR LOGIN NODE ---
import unittest.mock as mock
sys.modules['pyrender'] = mock.MagicMock()
sys.modules['OpenGL'] = mock.MagicMock()
sys.modules['OpenGL.GL'] = mock.MagicMock()
sys.modules['OpenGL.platform'] = mock.MagicMock()
# ----------------------------------------------

import torch
from datasets.annyone import AnnyOne

# ---- everything goes into this buffer, then written to a txt file ----
LOG_PATH = os.environ.get('SHAPE_LOG', '/netscratch/najib/multi-hmr/shape_report.txt')
report = []
def R(*a):
    line = " ".join(str(x) for x in a)
    report.append(line)  # buffered only; not printed to console

R("Anny shape inspection report")
R("generated:", datetime.datetime.now().isoformat())
R("=" * 60)

R("\nLoading AnnyOne Dataset...")
dataset = AnnyOne(data_folder='/netscratch/najib/anydataset/', img_size=672)
R("dataset size:", len(dataset))

# ============================================================
# PART 1: raw per-sample inspection (before collate)
# ============================================================
img, annot = dataset[0]
R("\n================= PART 1: RAW SAMPLE =================")
R("annot keys:", list(annot.keys()))

humans = annot['humans']
R("n humans in sample 0:", len(humans))

h0 = humans[0]
if isinstance(h0, dict):
    R("human[0] keys:", list(h0.keys()))
    for k, v in h0.items():
        if hasattr(v, 'shape'):
            R(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        elif isinstance(v, (list, tuple)):
            R(f"  {k}: {type(v).__name__} of length {len(v)}")
        else:
            R(f"  {k}: {type(v).__name__} = {v}")

    for cand in ['anny_shape', 'shape', 'phenotype', 'phenotypes', 'betas']:
        if cand in h0:
            s = torch.as_tensor(h0[cand]).float().reshape(-1)
            R(f"\n>>> FOUND '{cand}': len={s.numel()}, "
              f"min={s.min():.4f}, max={s.max():.4f}")
            R(">>> values:", [round(x, 4) for x in s.tolist()])
else:
    R("human[0] is not a dict, it is:", type(h0))

# ============================================================
# PART 2: batched inspection (what prepare_gt actually sees)
# ============================================================
R("\n================= PART 2: AFTER COLLATE (authoritative) =================")
try:
    from datasets.bedlam import collate_fn
    img_b, y = collate_fn([dataset[0]])
    R("batched keys:", list(y.keys()))
    if 'anny_shape' in y:
        s = y['anny_shape']
        R(f"anny_shape: shape={tuple(s.shape)}, dtype={s.dtype}, "
          f"min={float(s.min()):.4f}, max={float(s.max()):.4f}")
        flat = s.reshape(-1, s.shape[-1])
        R("first human values:", [round(x, 4) for x in flat[0].tolist()])
    else:
        R("!! 'anny_shape' not in batched keys — check collate_fn")
except Exception as e:
    import traceback
    R("Collate step failed (fine on login node, PART 1 is enough):")
    R(traceback.format_exc())

# ============================================================
# PART 3: what the model expects
# ============================================================
R("\n================= PART 3: ANNY PHENOTYPE LABELS =================")
try:
    import anny
    body = anny.create_fullbody_model(
        remove_unattached_vertices=False,
        all_phenotypes=True,
    )
    labels = list(body.phenotype_labels)
    R("phenotype_labels:", labels)
    R("n_labels:", len(labels))
    used = ['age', 'gender', 'weight', 'height', 'muscle', 'proportions']
    R("indices used by the model:",
      {k: labels.index(k) for k in used if k in labels})
    R("used keys NOT in labels:", [k for k in used if k not in labels])
except Exception as e:
    import traceback
    R("Could not build anny body model:")
    R(traceback.format_exc())

# ============================================================
# WRITE THE REPORT TO A TXT FILE
# ============================================================
try:
    with open(LOG_PATH, 'w') as f:
        f.write("\n".join(report) + "\n")
    pass  # silent: everything is in LOG_PATH
except Exception as e:
    import traceback
    try:
        with open(LOG_PATH + ".error", "w") as ef:
            ef.write(traceback.format_exc())
    except Exception:
        pass