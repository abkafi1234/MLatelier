# Installation

## Requirements

- Python 3.9 – 3.14
- Linux, macOS, or Windows
- ~4 GB free disk space (PyTorch and the transformer models dominate this)
- A CUDA-capable GPU is **optional**. It benefits only the Vision DL tab and the
  HuggingFace fine-tuning track. All tabular work and the TF-IDF NLP track run
  perfectly well on CPU.

## Standard install

```bash
pip install mlatelier
mlatelier
```

The second command starts the dashboard and opens `http://localhost:8501`.

## No optional extras

That single install gives you every feature: tabular, vision, NLP, all the
explainers, ONNX export, and counterfactuals. There is deliberately no
`pip install "mlatelier[something]"` for runtime features — the dashboard is
used by people who do not write code, and a feature that fails mid-session with
"install another package first" is a bug, not a configuration choice.

The one extra is for contributors and installs no runtime code:

```bash
pip install "mlatelier[dev]"    # pytest, pytest-cov, build, onnxruntime
```

## GPU support

Install a CUDA build of PyTorch **before** installing MLatelier, so that pip does
not pull the CPU-only wheel first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mlatelier
```

Verify what the app detected:

```python
from mlatelier.vision_engine import get_device_info
print(get_device_info())
```

## Install from source

```bash
git clone https://github.com/abkafi1234/MLatelier.git
cd MLatelier
pip install -e ".[dev]"
pytest
```

## Troubleshooting

**`mlatelier: command not found`**
The console script landed in a directory that is not on your `PATH`. Run it via
the module instead:

```bash
python -m mlatelier
```

**NLTK stopwords fail to download**
The TF-IDF track downloads the NLTK stopword corpus on first use. On an offline
or proxied machine, fetch it once manually:

```python
import nltk; nltk.download("stopwords"); nltk.download("punkt")
```

Alternatively, untick "Remove stopwords" in the NLP tab — the pipeline runs
without it.

**Port 8501 already in use**

```bash
streamlit run $(python -c "import mlatelier,os;print(os.path.join(os.path.dirname(mlatelier.__file__),'app.py'))") --server.port 8502
```

**Out-of-memory during vision training**
Lower the batch size in the Vision DL tab, or choose a smaller architecture
(MobileNetV3 rather than a ResNet or ViT). Vision training on CPU is feasible
for small datasets but slow.

**`ImportError` for xgboost / lightgbm / catboost**
These are declared dependencies, so a normal install includes them. If one is
missing in your environment, MLatelier drops that model from the catalogue
automatically rather than failing — the rest of the run is unaffected.

**The AI Assistant tab says no API key is configured**
This is expected. The assistant is inert until you supply a Google Gemini API
key in the tab. Nothing is transmitted anywhere until you do, and no other
feature depends on it.
