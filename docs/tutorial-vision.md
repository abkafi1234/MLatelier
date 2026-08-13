# Tutorial: Vision DL

Transfer learning over twelve pre-trained architectures, with Grad-CAM to check
what the network actually looked at.

## Dataset layout

One directory per class:

```
my_dataset/
├── cat/
│   ├── 001.jpg
│   └── 002.jpg
├── dog/
│   ├── 001.jpg
│   └── 002.jpg
└── rabbit/
    └── ...
```

Supply the folder path, or upload a ZIP with the same structure. Class names come
from the directory names. The split is 60/20/20 train/validation/test,
stratified by class.

Before training, the summary panel shows per-class counts and sample images.
Look at it — mislabelled folders and a class with eleven images are both obvious
here and invisible later.

## Architectures

Twelve pre-trained networks are available. **Use these exact registry keys** —
an unknown name raises `ValueError`:

| Tier | Registry keys | When |
|---|---|---|
| Light | `MobileNet_v3`, `EfficientNet_B0` | CPU-only, or a fast first pass |
| Standard | `ResNet18`, `ResNet50`, `DenseNet121`, `VGG16`, `EfficientNet_V2_S` | The sensible default |
| Modern conv | `ConvNeXt_T`, `ConvNeXt_S` | Strong accuracy, still Grad-CAM-able |
| Transformer | `ViT_B_16`, `Swin_T`, `Swin_S` | Large datasets, GPU, maximum accuracy |

On CPU, start with `MobileNet_v3` and a frozen backbone. A ViT on CPU is not a
practical experiment.

**Grad-CAM works on the nine convolutional architectures only.** `ViT_B_16`,
`Swin_T`, and `Swin_S` have no final convolutional feature map to attribute
over, so if visual explanation matters to you, pick a convolutional backbone.

## Freezing strategy

**`head_only`** replaces and trains just the classification head, leaving
pre-trained features untouched. Fast, low memory, and the right choice for small
datasets — a few hundred images per class cannot support updating millions of
backbone parameters without overfitting.

**Full fine-tuning** updates everything. Needs more data, more time, and a lower
learning rate, but wins when your domain is far from ImageNet (medical imaging,
satellite, microscopy).

## Training

The baseline runs fixed hyperparameters (`lr=1e-3`, `wd=0`) for a few epochs to
establish a reference. Optimisation then runs Bayesian search over learning rate
and weight decay, with cosine annealing and early stopping.

| Setting | Default | Notes |
|---|---|---|
| Epochs | 10 | 15–30 for full fine-tuning |
| Batch size | 32 | Halve it on out-of-memory; 8–16 is normal on 6 GB |
| Augmentation | on | Flips, rotations, colour jitter. Leave on unless orientation is semantically meaningful |
| Early stopping patience | 5 | Lower to 3 for quick exploration |
| LR scheduler | cosine | |

Augmentation deserves one thought rather than none: horizontal flips are wrong
for anything where left and right differ — text, dial gauges, some medical
views.

## Checkpointing

Vision training is the slowest thing in MLatelier, so *Checkpoint each epoch* is
on by default. After every epoch the model, optimiser, and scheduler state go to
`~/MLatelier/checkpoints/`. If the run is interrupted, re-running the same
configuration resumes from the last completed epoch.

Three things worth knowing:

- **The configuration is fingerprinted.** Architecture, learning rate, weight
  decay, epochs, freeze strategy, scheduler, and seed all feed the hash. Change
  any of them and training restarts clean — a resume can never silently continue
  a different experiment.
- **Optimiser state is saved, not just weights.** Resuming with a fresh Adam
  would reset the moment estimates and produce a visible bump in the loss curve.
- **Checkpoints are deleted on success.** They only persist after an abnormal
  exit, so the directory stays empty in normal use.

The three long phases — per-model optimisation, final test training, and the
winner's retrain — checkpoint separately and cannot overwrite each other.
Progress *through* Bayesian-optimisation trials is not resumable; individual
trials are short by design.

To inspect or clear checkpoints manually:

```python
from mlatelier.checkpoint import list_checkpoints, purge_checkpoints

for c in list_checkpoints():
    print(c["name"], "epoch", c["epoch"], f"{c['size_mb']} MB")

purge_checkpoints()
```

## Grad-CAM

Grad-CAM produces a heat map of the image regions that drove the prediction. The
UI applies it to misclassified samples by default, which is the highest-value
place to look.

What you are checking for is whether the network learned the object or the
context. A classifier that reaches 97 % by attending to the watermark in the
corner of every image from one class will collapse on new data, and the accuracy
number alone will never tell you. Grad-CAM will.

If the heat map sits on the background across many samples, the fix is data —
more varied backgrounds, tighter crops — not more epochs.

## Export and inference

Models export as `.pt`. Load them in the **Predict / Inference** tab and score
single images, a batch, or a whole folder.

## Scripting it

```python
from mlatelier.vision_engine import (
    run_vision_baseline, run_vision_optimization, get_device_info,
)

print(get_device_info())

baseline, class_names = run_vision_baseline(
    data_dir="my_dataset",
    selected_models=["ResNet18", "MobileNet_v3"],
    batch_size=32, freeze_strategy="head_only", baseline_epochs=3,
)

results = run_vision_optimization(
    data_dir="my_dataset",
    selected_models=["ResNet18", "MobileNet_v3"],
    baseline_results=baseline,
    epochs=10, n_bo_calls=20, export_model=True,
)
```

## Troubleshooting

**CUDA out of memory.** Halve the batch size; then choose a lighter
architecture. Restart the app to clear cached allocations.

**Training accuracy climbs, validation does not.** Overfitting. Freeze the
backbone, add augmentation, or get more data — in that order of cost.

**Every prediction is one class.** Severe class imbalance. Check the class
distribution panel; enable imbalance handling.

**It is unbearably slow.** Confirm you are on GPU with `get_device_info()`. A
CPU fallback is often a PyTorch install that pulled the CPU-only wheel — see
[Installation](installation.md#gpu-support).
