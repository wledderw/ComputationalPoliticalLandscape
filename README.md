# Computational Political Landscape of the Netherlands and Prime Minister Schoof’s Position

## Introduction

This repository hosts the code used for getting the results as in the paper Computational Political Landscape of the Netherlands and Prime Minister Schoof’s Position, by Wessel Ledder and Iris Hendrickx, 2026.

## Contents

### Repository structure

This repository should have the following structure:

```
ComputationalPoliticalLandscape
│   bert_finetune.py
│   inference.ipynb
│   LICENSE
│   README.md
│   requirements.txt
│   scrape_data.py
│   split_data.py
│   .gitignore
│
├───data
│       data.csv
│       schoof.csv
│       test.csv
│       train.csv
│       val.csv
│
├───liwc_data
│       highest_second_factor.csv
│       highest_second_factor_liwc.csv
│       lowest_second_factor.csv
│       lowest_second_factor_liwc.csv
│    
├───plots
│       boxplot_liwc.png
│       computational_political_landscape_test_set.png
│       computational_political_landscape_train_test_set.png
│       distance_matrix.png
│       samples_lengths.png
│       samples_parties_counts.png
│       separate_speeches_test_set.png
│
└───weights
    │   README.md
    │   ### DOWNLOAD THE FOLLOWING FOLDER, SEE weights/README.md:
    └───params_epoch_19_bs32_lr2e-05_hiddendropout0.5_attentiondropout0.5_weightdecay0.0003_warmup10
            config.json
            model.safetensors
```

### Data

Data was scraped from the 'Plenaire Verslagen' of the House of Representatives (Tweede Kamer) (https://www.tweedekamer.nl/kamerstukken/plenaire_verslagen).

### Requirements

The requirements can be found in the file `requirements.txt`. To make a Python virtual environment with the necessary packages, run the following commands:

```bash
git clone https://github.com/wledderw/ComputationalPoliticalLandscape.git  # to get scripts and data
python3 -m venv .venv
source /venv/bin/activate  # for Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

### Code

The data has already been scraped and splitted over datasets, and saved in the folder `data`.

However, if you want to re-scrape and re-split the data, run:

```bash
python3 scrape_data.py
python3 split_data.py
```

Model weights have to be downloaded from HuggingFace Hub. See weights/README.md for more specifics.

However, if you want to fine-tune the BERTje model yourself and get the weights, please run:

```bash
python3 bert_finetune.py
```

The weigths will be saved in the folder called `weights`.

Finally, to create the political landscape, run the command:

```bash
jupyter notebook
```

And open the Jupyter Notebook called `inference.ipynb` from here, and run all cells.

All plots made during these scripts and notebooks are saved in the folder `plots`.

## Citation

If you find this repository helpful in an academic setting, please cite:

TBA