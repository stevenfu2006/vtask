# VTASK Training

GRPO fine-tuning scripts for training language models on VTASK reasoning domains.

## Setup

```bash
pip install -e ..   # install VTASK from the parent directory
pip install -r requirements.txt
```

GPU with ≥16 GB VRAM recommended. For multi-GPU, prefix commands with `accelerate launch`.

## Training

Fine-tune a model on a single domain:

```bash
python train.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --domain scheduling \
  --steps 200 \
  --output-dir ./output/scheduling
```

Train on all 6 domains simultaneously:

```bash
python train.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --domain all \
  --steps 500 \
  --dataset-size 3000 \
  --output-dir ./output/mixed
```

Fix difficulty level for curriculum learning:

```bash
# Stage 1: easy tasks only
python train.py --model ... --difficulty 1 --steps 100 --output-dir ./output/stage1

# Stage 2: medium tasks, warm-starting from stage 1
python train.py --model ./output/stage1 --difficulty 3 --steps 200 --output-dir ./output/stage2
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model or checkpoint path |
| `--domain` | `scheduling` | Domain name or `all` |
| `--difficulty` | mixed | Fix difficulty 1-5 |
| `--steps` | `200` | Training steps |
| `--batch-size` | `4` | Per-device batch size |
| `--dataset-size` | `500` | Tasks to generate |
| `--learning-rate` | `5e-6` | Learning rate |
| `--output-dir` | `./output` | Checkpoint save path |
| `--no-wandb` | off | Disable W&B logging |

## Evaluation

Compare base vs fine-tuned accuracy:

```bash
python eval_model.py \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --finetuned ./output/scheduling \
  --domain scheduling \
  --size 100
```

Evaluate across all domains:

```bash
python eval_model.py \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --finetuned ./output/mixed \
  --domain all \
  --size 50
```

## How rewards work

The `RewardFunction` in `VTASK/training.py` extracts the model's answer from
`<answer>...</answer>` tags and passes it to the domain's `score_answer()` verifier.
No LLM judge is involved — all rewards are computed algorithmically.

The system prompt instructs the model to wrap its final answer in tags:

```
You are solving reasoning problems. Think through the problem carefully,
then provide your final answer. Wrap your final answer in <answer> and
</answer> tags. Example: <answer>42</answer>
```

## Using VTASK training utilities directly

```python
import VTASK
from VTASK.training import RewardFunction, VTASKDataset, format_chat_prompt

# Build a dataset
generator = VTASK.registry.REGISTRY["scheduling"]()
task_ds = generator.create_dataset(size=500, seed=42)

reward_fn = RewardFunction(generator)
vtask_ds = VTASKDataset(task_ds, reward_fn)

# Get a HuggingFace dataset for TRL
hf_dataset = vtask_ds.to_hf_dataset()

# Use reward_fn directly
score = reward_fn(
    completions=["<answer>14</answer>"],
    prompts=[vtask_ds[0]["prompt_str"]],
)
```
