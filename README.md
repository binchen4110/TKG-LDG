# (TKG-LDG) Enhancing Temporal Knowledge Graph for Future Event Prediction with Long-Term Dense Graph

Implementation of **TKG-LDG** for the paper: Enhancing Temporal Knowledge Graph for Future Event Prediction with Long-Term Dense Graph

# Code Structure

```sh
TKG-LDG-release
├─ data
│  ├─ ICEWS18
│  ├─ ICEWS05-15
│  ├─ ICEWS14s
│  ├─ ICEWS14
│  ├─ GDELT
│  ├─ WIKI
│  └─ YAGO
├─ configs
│  ├─ ICEWS18.json
│  ├─ ICEWS05-15.json
│  ├─ ICEWS14.json
│  ├─ GDELT.json
│  ├─ WIKI.json
│  └─ YAGO.json
├─ tkg_ldg
│  ├─ __init__.py
│  ├─ data.py
│  ├─ layers.py
│  ├─ model.py
│  ├─ decoder.py
│  ├─ metrics.py
│  └─ train.py
├─ scripts
│  ├─ run_all_datasets.py
│  └─ summarize.py
├─ requirements.txt
├─ README.md
└─ main.py
```

## Dependency
Install the dependencies in a Python 3.10 environment:

```sh
pip install -r requirements.txt
```

# Train

Train TKG-LDG on each dataset using the commands below. These commands use the selected two-channel decoder with λ=0.7.


**ICEWS18**

```sh
nohup python -u main.py --config configs/ICEWS18.json --data-root data --device cuda:0 --seed 2024 --epochs 40 --patience 3 --entity-weight 0.7 --output runs/ICEWS18 > icews18.txt
```

**ICEWS05-15**

```sh
nohup python -u main.py --config configs/ICEWS05-15.json --data-root data --device cuda:0 --seed 2024 --epochs 40 --patience 3 --entity-weight 0.7 --output runs/ICEWS05-15 > icews05-15.txt
```

**ICEWS14**

```sh
nohup python -u main.py --config configs/ICEWS14.json --data-root data --device cuda:0 --seed 2024 --epochs 40 --patience 3 --entity-weight 0.7 --output runs/ICEWS14 > icews14.txt
```

**GDELT**

```sh
nohup python -u main.py --config configs/GDELT.json --data-root data --device cuda:0 --seed 2024 --epochs 40 --patience 3 --entity-weight 0.7 --output runs/GDELT > gdelt.txt
```

**WIKI**

```sh
nohup python -u main.py --config configs/WIKI.json --data-root data --device cuda:0 --seed 2024 --epochs 40 --patience 3 --entity-weight 0.7 --output runs/WIKI > wiki.txt
```

**YAGO**

```sh
nohup python -u main.py --config configs/YAGO.json --data-root data --device cuda:0 --seed 2024 --epochs 40 --patience 3 --entity-weight 0.7 --output runs/YAGO > yago.txt
```
