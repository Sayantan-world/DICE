# Data

Download from [chocokiddo/DICE-Bench](https://huggingface.co/datasets/chocokiddo/DICE-Bench).

## Evaluation triplets (`default` config)

| Split | Size | Use |
|---|---|---|
| `triplets_250` | 250 | Primary evaluation split (`seed=42`) for reported tutoring tables |
| `triplets_all` | 10,273 | Full easy / target / hard corpus |

```bash
python scripts/download_data.py --split triplets
```

Writes `data/triplets_250.json` and `data/triplets_all.json`. Experiment runners download `triplets_250` automatically if it is missing.

```python
from datasets import load_dataset
eval_split = load_dataset("chocokiddo/DICE-Bench", split="triplets_250")
```

## Policy annotations (`policy` config)

20,477 cleaned turn-level action labels, grouped by source problem ID (`seed=42`).

| Split | Rows |
|---|---|
| `train` | 16,383 |
| `validation` | 2,048 |
| `test` | 2,046 |

```bash
python scripts/download_data.py --split policy
```

Writes `data/policy/{train,val,test,all}.jsonl`. Policy training and eval runners download this automatically.

```python
from datasets import load_dataset
policy = load_dataset("chocokiddo/DICE-Bench", "policy")
```

## Intervention values (`iv` config)

| Split | Rows | Use |
|---|---|---|
| `target_observed` | 6,091 | Target-dialogue IV used for policy weights and the paper IV audit |
| `easy_synthetic` | 479 | Easy step-variant IV |
| `hard_synthetic` | 809 | Hard step-variant IV |

```bash
python scripts/download_data.py --split iv
```

Join to policy rows with `_row_id`.