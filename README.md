# Repository structure:

master: Contains end-results + paper source
A100, GTX1650, T4: Contains hardware-specific benchmarks

# Setup
```bash
git clone https://github.com/saifmb0/sparse-vits.git
cd sparse-vits
git checkout -b branch_name
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```
# Quickstart
```bash
.venv/bin/python -m run_all.py \
      --bench 1..6 # optional
```
