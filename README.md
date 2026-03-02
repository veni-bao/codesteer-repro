# CodeSteer Reproduction

A minimal, self-contained reproduction of [CodeSteer (ICML 2025)](https://arxiv.org/abs/2502.04350) by Yongchao Chen et al.

## What This Is

CodeSteer guides a large TaskLLM (e.g., GPT-4o or MiniMax) between **code generation** and **textual reasoning** using a fine-tuned small model (Llama-3.1-8B: `yongchao98/CodeSteer-v1.0`).

This repo reproduces the core inference pipeline on a subset of SymBench tasks and plots results.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp config.example.yaml config.yaml
# Edit config.yaml and fill in your API key + model settings
```

### 3. Run evaluation

```bash
# Run on a small subset (5 tasks, 10 samples each) — fast demo
python run_eval.py --tasks game24 sudoku path_plan logical_deduction number_multiply \
                   --n_samples 10 --output results/demo_run.json

# Run full SymBench (37 tasks, 100 samples) — needs GPU + budget
python run_eval.py --all_tasks --n_samples 100 --output results/full_run.json
```

### 4. Plot results

```bash
python plot_results.py --input results/demo_run.json --output figures/
```

## Hardware Requirements

| Mode | GPU VRAM | Notes |
|------|----------|-------|
| Full CodeSteerLLM | 16 GB | Llama-3.1-8B fp16 |
| Quantized (4-bit) | 8 GB | Use `--quantize 4bit` flag |
| API-only (no local GPU) | 0 GB | Use `--codesteer_mode api` with OpenAI/MiniMax |

## Repository Structure

```
codesteer-repro/
├── config.example.yaml      # Template config (fill in API keys)
├── requirements.txt         # Python dependencies
├── run_eval.py              # Main evaluation script
├── plot_results.py          # Visualization
├── codesteer/
│   ├── __init__.py
│   ├── steer_llm.py         # CodeSteerLLM wrapper (local or API)
│   ├── task_llm.py          # TaskLLM wrapper (OpenAI / MiniMax / etc.)
│   ├── checkers.py          # Symbolic Checker + Self-answer Checker
│   ├── pipeline.py          # Multi-turn CodeSteer pipeline
│   └── tasks/               # SymBench task definitions
│       ├── __init__.py
│       ├── game24.py
│       ├── sudoku.py
│       ├── path_plan.py
│       ├── logical_deduction.py
│       └── number_multiply.py
└── results/                 # Output JSON files (gitignored)
```

## Baseline Comparison

The script also runs baseline methods for comparison:
- Only Question (raw prompt)
- All Text + CoT
- All Code + CoT
- CodeSteer (this paper)

Results are plotted as bar charts matching Figure 1 / Table 1 in the paper.
