# TCKGE Reproducibility

The workflow is organized into three Python entry points:

1. `phase1.py` trains or refreshes embedding matrices.
2. `phase2.py` runs tolerance extraction, semantic purity/coherence evaluation, baselines, and diagnostics.
3. `phase3.py` regenerates some figures and tables.

## Directory Layout

- `phase1.py`, `phase2.py`, `phase3.py`: main reproducibility workflow.
- `requirements.txt`: Python package requirements.

## Environment Setup

Recommended environment:

- Python 3.10 or 3.11.
- CUDA-capable PyTorch for full training runs. CPU execution is possible for many analysis steps, but full training and baselines can be very slow.

Install dependencies from the package root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
