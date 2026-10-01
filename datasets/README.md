# datasets/ — dataloader package (MISSING FROM THIS REPO)

`train.py`, `test_loader.py`, `check_shape.py` and `explore_anny.py` import:

```
from datasets.annyone import AnnyOne
from datasets.bedlam import BEDLAM, collate_fn
from datasets.threedpw import THREEDPW
from datasets.ehf import EHF
```

Until 2026-10 this folder was listed in `.gitignore`, so the source only exists on
the cluster. Copy it in and commit it:

```
cp -r /netscratch/najib/multi-hmr/datasets/*.py datasets/
git add datasets/ && git commit -m "Add dataloader package"
```

Only the `.py` files belong here. Data itself stays under `/netscratch/najib/anydataset/`
and `/ds-av/public_datasets/3DPW/`, which `train.py` reads via `--data_dir`-style
defaults. Nothing in this repository can load data until this folder is populated.
