
# Explainable Archive Feedback and Quality-Diversity Search for Automated Design of Differential Evolution Algorithms
# Features

* Surrogate-assisted evolutionary search
* MAP-Elites quality-diversity archive
* Differential Evolution (DE)-constrained code generation
* XGBoost surrogate model
* TreeSHAP-based surrogate interpretation
* Automatic evaluation on MA-BBOB benchmark
* Parallel execution using Joblib
* Local LLM inference through Ollama
* Fully reproducible experimental pipeline

---

# Repository Structure

```text
.
├── configs/                # Experiment configurations
├── prompts/                # DE-constrained mutation prompts
├── llamea/                 # Evolutionary framework
├── benchmark/              # Benchmark interfaces
├── scripts/                # Experiment scripts
├── logs/                   # Execution logs
├── results/                # Generated results
├── analysis/               # Statistical analysis
├── figures/                # Paper figures
├── models/                 # Surrogate models
└── README.md
```

---

# System Requirements

## Hardware

* Multi-core CPU recommended
* 32 GB RAM or higher recommended
* NVIDIA GPU recommended for local LLM inference

## Operating System

* Windows 11 Pro
* Linux (Ubuntu 22.04 or newer) should also work

---

# Software Requirements

* Python 3.14 (Framework requires Python ≥ 3.11)
* Ollama
* Git

---

# Python Libraries

Major dependencies include:

* iohblade 0.2.3
* llamea 1.2.0
* ioh 0.3.22
* xgboost 2.1.4
* numpy 2.x
* pandas
* scikit-learn
* joblib
* shap
* matplotlib

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

# Installing Ollama

Install Ollama from the official website.

After installation, pull the required language model:

```bash
ollama pull qwen3-coder:30b
```

Verify installation:

```bash
ollama list
```

Start the Ollama server:

```bash
ollama serve
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd SAGE-QD-DE
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows

```bash
.venv\Scripts\activate
```

Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Experimental Configuration

The default experimental configuration is:

| Parameter                |                    Value |
| ------------------------ | -----------------------: |
| Parent population (μ)    |                        6 |
| Offspring population (λ) |                       18 |
| Survivor selection       |            μ + λ elitism |
| Parent selection         |       Tournament (k = 3) |
| Mutation prompts         | 3 DE-constrained prompts |
| Evaluation budget        |      204 LLM evaluations |
| Independent runs         |                        5 |
| MAP-Elites descriptor    |            4-dimensional |
| Archive resolution       |            4 × 4 × 4 × 4 |
| Surrogate model          |                  XGBoost |
| Maximum tree depth       |                        4 |
| Learning rate            |                      0.1 |
| Attribution              |                 TreeSHAP |

---

# Benchmark Configuration

The framework evaluates generated optimizers on the MA-BBOB benchmark.

| Parameter          |                       Value |
| ------------------ | --------------------------: |
| Dimension          |                          10 |
| Training instances |                         0–9 |
| Test instances     |                      20–119 |
| Evaluation budget  | 50,000 function evaluations |
| Search space       |                    [-5,5]^d |
| Performance metric |                        AOCC |

---

# Running Experiments

Run the default experiment:

```bash
python main.py
```

To use a custom configuration:

```bash
python main.py --config configs/<config_file>.yaml
```

---

# Parallel Execution

The framework supports parallel evaluation using Joblib.

Default configuration:

* Five concurrent evaluations
* Candidate timeout: 600 seconds

---

# Outputs

The framework automatically generates:

* Generated DE algorithms
* MAP-Elites archive
* Performance logs
* AOCC scores
* Surrogate models
* TreeSHAP explanations
* Statistical summaries
* Figures for publication

---

# Citation

If you use this repository in your research, please cite the accompanying publication.

```bibtex
@article{YOUR_PAPER,
  title={Surrogate-Assisted Quality-Diversity Differential Evolution},
  author={...},
  journal={...},
  year={2026}
}
```

---

# License

Specify the repository license (e.g., MIT, Apache-2.0, GPL-3.0).

---

# Contact

For questions, bug reports, or research collaborations, please open an issue or submit a pull request through GitHub.
