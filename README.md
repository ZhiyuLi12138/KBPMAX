# KBPMAX

G4BP greedy algorithm for k-BP function maximization, as described in the paper "K-BP-FUNCTION".

## Installation

Clone the repository and install locally with pip:

```bash
git clone https://github.com/ZhiyuLi12138/KBPMAX.git
cd KBPMAX
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

## Usage

### As a library

```python
from kbpmax import KBPFunction, G4BP

func = KBPFunction(n=10, k=3, alpha=0.3, beta=0.3, lambda_param=0.5, C=5)
alg = G4BP(func, constraint_type="total", C=5)
x_hat, h_value = alg.run()
print(f"Solution value: {h_value:.4f}")
```

### Command-line interface

After installation, run experiments directly:

```bash
# Full run (validation + parameter sweep)
kbpmax

# Small-instance validation only
kbpmax --mode validate

# Full parameter sweep with custom settings
kbpmax --mode experiment --n 20 --C 10 --k 3

# Or run as a script without installing
python kbpmax.py --mode validate
```