# Downloading the weights

Instead of re-fine-tuning BERTje, you can also download the model weights.

## Option 1: download via browser

Go to https://huggingface.co/wledder/ComputationalPoliticalLandscape

Download the folder called `params_epoch_19_bs32_lr2e-05_hiddendropout0.5_attentiondropout0.5_weightdecay0.0003_warmup10`

Place this folder in the `weights` folder in the repository, as specified in the file tree in the `README.md`.

## Option 2: download via HuggingFace Hub

First, install HuggingFace Hub in your virtual environment:

```bash
pip install huggingface_hub
```

Then, run the following code in Python:

```py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="wledder/ComputationalPoliticalLandscape",
    allow_patterns="params_epoch_19_bs32_lr2e-05_hiddendropout0.5_attentiondropout0.5_weightdecay0.0003_warmup10/*",
    cache_dir="weights"
)
```

Ensure that the file structure is consistent with the file tree in `README.md`.