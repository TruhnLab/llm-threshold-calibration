# Single-Pass Decision Scores Enable Rapid Classification with Adjustable Thresholds in Medical Multimodal Large Language Models

[![Python](https://img.shields.io/badge/Python-3.12.3-orange.svg)](https://python.org)
[![vLLM](https://img.shields.io/badge/vLLM-0.9.2-green.svg)](https://docs.vllm.ai/en/v0.9.2/)

This repository contains the implementation and evaluation code for the paper _"Single-Pass Decision Scores Enable Rapid Classification with Adjustable Thresholds in Medical Multimodal Large Language Models"_.

A multimodal large language model is asked a yes/no question about an image (e.g. _"Does the imaging show cardiomegaly?"_). Instead of reading the generated answer, we take the probabilities of the tokens "yes" and "no" from a single forward pass and normalize them to a continuous decision score. Like the output of a conventional classifier, this score can be thresholded at any operating point. We compare it with the raw (unnormalized) "yes" probability and with a probability that the model states as text.

## Requirements
**Note:** we used Linux, Python 3.12.3 and [`uv`](https://docs.astral.sh/uv/) (all packages are pinned in [`requirements.txt`](requirements.txt)). Inference was run on a single NVIDIA H100 GPU. The evaluation runs on the CPU.

**Setup Instructions:**

1. Ensure [`uv`](https://docs.astral.sh/uv/getting-started/installation/) is installed on your system.

2. Create and activate the virtual environment (run all commands from the repository folder):
```
$ uv venv --python 3.12.3
$ source .venv/bin/activate
```

3. Install the required packages:
```
$ uv pip install -r requirements.txt
```

The environment setup is now complete.

## Quick Evaluation

The outputs of all inference runs (4 models × 6 datasets) are part of this repository, so the results of the paper can be reproduced without a GPU and without downloading any dataset:

- [`output`](output/): raw model outputs per model and dataset (`eval_<model>_<dataset>.pkl`): the log-probabilities of "yes" and "no" and the text-elicited probability for every image and label, for the validation and the test split.
- [`evalOutput`](evalOutput/): evaluation results per model and dataset (`eval_<model>_<dataset>_results.csv`): AUROC, accuracy, F1, precision, sensitivity and specificity with 95% bootstrap confidence intervals, optimal thresholds and p-values.

To recreate the tables and figures of the paper from these files, run:
```
$ python3 tableCreator.py > results_tables.txt
$ python3 supplementTables.py
$ python3 plotThresholds.py
```

- [`tableCreator.py`](tableCreator.py) prints the rows of Tables 1–3 of the paper (LaTeX). Expected output: [`results_tables.txt`](results_tables.txt)
- [`supplementTables.py`](supplementTables.py) writes the prompt, label and per-class AUROC tables of the Supplementary Information to `evalOutput/supplementTables.tex`
- [`plotThresholds.py`](plotThresholds.py) plots the optimal thresholds of MedGemma 27B (Fig. 5 of the paper) to `evalOutput/eval_MedGemma27B_<dataset>/`

### Recomputing the evaluation

[`eval.py`](eval.py) computes all metrics from the model outputs in [`output`](output/) and additionally plots ROC curves and AUROC bar plots (Figs. 3 and 4 of the paper) as SVG files into the same folders (these plots are not part of the repository). Results that already exist in [`evalOutput`](evalOutput/) are skipped, so delete them first. The bootstrap is seeded, so the recomputed CSV files are identical to the provided ones.

To recompute a single model and dataset (here MedGemma 27B on G1020, about 2 minutes on 32 CPU cores):
```
$ rm -r evalOutput/eval_MedGemma27B_G1020
$ python3 eval.py
$ python3 plotThresholds.py
$ git status evalOutput/eval_MedGemma27B_G1020   # only new plots, the results CSV is reproduced exactly
```

To recompute everything (all CPU cores, 10,000 bootstrap samples per metric, several hours):
```
$ rm -r evalOutput
$ python3 eval.py
$ python3 tableCreator.py > results_tables.txt
$ python3 supplementTables.py
$ python3 plotThresholds.py
```

## Used Datasets

All datasets are publicly available. Download them from the original sources and copy them into a folder `data/` in this repository (symbolic links work as well):

```
data/
├── CheXpert/            # created by preprocessCheXpert.py (see below)
├── MIMIC-CXR-JPG/
├── HAM10000/
├── DERM12345/
├── NCT-CRC-HE-100K/
└── G1020/
```

The paths are set at the top of each dataset file (e.g. `CHEXPERT_BASE_PATH` in [`ChexpertDataset.py`](ChexpertDataset.py)). The images of the validation and test splits of HAM10000, DERM12345, NCT-CRC-HE-100K and G1020 are listed in [`splits`](splits/) for reference.

- **CheXpert** [1]: chest radiographs, 14 findings; official validation set (234 images) and test set (668 images)
  - The validation and the labelled test set of CheXpert are both part of [CheXlocalize](https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c) [2] (the full [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) is not needed). Copy the folder `CheXpert` of the download (containing `val/`, `val_labels.csv`, `test/` and `test_labels.csv`) to `data/chexlocalize/CheXpert/`
  - Crop the images to a square and downscale them to 1024×1024 pixels:
    ```
    $ python3 preprocessCheXpert.py
    ```
    This creates `data/CheXpert/` with `valid.csv`, `valid/`, `test_labels.csv` and `test/`. The folder `data/chexlocalize/` is not needed afterwards.

- **MIMIC-CXR-JPG v2.0.0** [3]: chest radiographs, 14 findings; official validation split (1,808 studies) and test split (3,269 studies)
  - Credentialed access via [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) (v2.1.0 contains the same files and can be used as well)
  - Copy the dataset to `data/MIMIC-CXR-JPG/` (containing `files/`) and unpack `mimic-cxr-2.0.0-split.csv.gz`, `mimic-cxr-2.0.0-chexpert.csv.gz` and `mimic-cxr-2.0.0-metadata.csv.gz` in this folder

- **HAM10000** [4]: dermoscopic images, 18 hierarchical diagnosis labels; 5,000 validation and 5,000 test images
  - Download the ISIC Archive version ([collection 212](https://api.isic-archive.com/collections/212/)) with images and metadata
  - Copy all images (`ISIC_<id>.jpg`) to `data/HAM10000/` and save the metadata file as `data/HAM10000/ham10000_metadata_2025-07-18.csv` (we downloaded it on 2025-07-18)
  - **Note:** the validation and test images of HAM10000 and DERM12345 are drawn by a seeded shuffle of the rows of the metadata file. A newer ISIC export with a different row order would result in different splits; compare the selected images with [`splits/HAM10000.csv`](splits/HAM10000.csv) and [`splits/DERM12345.csv`](splits/DERM12345.csv)

- **DERM12345** [5]: dermoscopic images, 37 hierarchical diagnosis labels; 5,000 validation and 5,000 test images
  - Download the dataset from the [ISIC Archive](https://api.isic-archive.com/doi/derm12345/) with images and metadata
  - Copy all images (`ISIC_<id>.jpg`) and `metadata.csv` to `data/DERM12345/`

- **NCT-CRC-HE-100K** [6]: histopathology patches of colorectal cancer, 9 tissue classes; 5,000 validation and 5,000 test patches
  - Download `NCT-CRC-HE-100K.zip` from [Zenodo](https://zenodo.org/records/1214456) and unpack it, so that `data/NCT-CRC-HE-100K/` contains the 9 class folders (`ADI`, `BACK`, `DEB`, `LYM`, `MUC`, `MUS`, `NORM`, `STR`, `TUM`)
  - In file names, this dataset is called `NCT_CRC_HE_100K_1000` (a historical name; 5,000 validation and 5,000 test patches are used)

- **G1020** [7]: fundus photographs, glaucoma vs. normal; 816 validation (`Training`) and 204 test (`Testing`) images
  - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sohamchakraborty2004/g1020-glaucoma) and copy it to `data/G1020/`, so that it contains `Training/` and `Testing/`, each with the subfolders `glaucoma/` and `normal/`

## Reproducing the Complete Results

**Note:** Inference on GPUs is not bit-exact, so a new inference run may yield slightly different results from those reported in the paper.

### Prerequisites
1. A GPU with enough memory for the chosen model (we used one NVIDIA H100 with 94 GB; the 4B models also run on smaller GPUs)
2. A [HuggingFace access token](https://huggingface.co/settings/tokens) with access to [MedGemma](https://huggingface.co/google/medgemma-27b-it) and [Gemma 3](https://huggingface.co/google/gemma-3-27b-it) (accept the license on the model pages)
3. All datasets in `data/` (see section _Used Datasets_)

### Steps

1. **Configure HuggingFace:**
   - Open [`VLLMModel.py`](VLLMModel.py)
   - Set `CACHE_PATH` to the folder where the model weights should be stored and `USED_TOKEN` to your HuggingFace token
   - The model weights (8–55 GB per model) are downloaded at the first run of each model

2. **Prepare the output folders:**
   - Delete the folders `output` and `evalOutput` to remove the data from our runs

3. **Run inference:**
   ```
   $ python3 runLLM.py MedGemma27B
   $ python3 runLLM.py MedGemma4B
   $ python3 runLLM.py Gemma27B
   $ python3 runLLM.py Gemma4B
   ```
   Each call runs one model on all six datasets. For every image and label, the model is queried once with the yes/no prompt (log-probabilities of "yes" and "no") and once with the JSON prompt (text-elicited probability, guided decoding). The results are saved as `output/eval_<model>_<dataset>.pkl`. The model revisions are pinned in [`runLLM.py`](runLLM.py).

4. **Run the evaluation:**
   ```
   $ python3 eval.py
   $ python3 tableCreator.py > results_tables.txt
   $ python3 supplementTables.py
   $ python3 plotThresholds.py
   ```
   See section _Quick Evaluation_ for details.

### Latency Benchmark

The latency of a single decision (one forward pass, one generated token) was measured on one NVIDIA H100 GPU. Our measurements are in [`benchmarkOut`](benchmarkOut/).
```
$ python3 benchmarkLatency.py MedGemma27B   # or MedGemma4B, Gemma27B, Gemma4B
```
The prompts of all datasets are taken from [`benchmarkPrompts.json`](benchmarkPrompts.json) (created from the model outputs with `python3 benchmarkLatency.py export`); the images are sampled from the datasets in `data/`.

## Methodology

1. **Querying** (code in `runLLM.py`, `VLLMModel.py`): every label is queried in a separate context with the image and a yes/no prompt. One forward pass yields the log-probabilities of the tokens "yes" and "no". A second query asks for the probability as JSON (guided decoding with xgrammar).
2. **Decision scores** (code in `eval.py`):
   - normalized: p(yes) / (p(yes) + p(no))
   - raw (unnormalized): p(yes)
   - text-elicited: the probability stated in the JSON answer
3. **Thresholds** (code in `eval.py`): metrics are computed on the test split at the default threshold of 0.5 and at the threshold that maximizes Youden's J on the validation split.
4. **Statistics** (code in `eval.py`): 95% confidence intervals from 10,000 bootstrap samples of the test split; the mean AUROC of the normalized score is compared with the other two scores by a paired bootstrap test.

## File Structure

```
├── README.md                    # This file
├── requirements.txt             # Python packages (uv pip install -r requirements.txt)
├── runLLM.py                    # Inference: all datasets for one model
├── VLLMModel.py                 # vLLM wrapper: generation and yes/no log-probabilities
├── ChexpertDataset.py           # Dataset loading (one file per dataset)
├── MIMIC_Dataset.py
├── HAM10000.py
├── DERM12345.py
├── NCT_CRC_HE_100K.py
├── G1020.py
├── preprocessCheXpert.py        # Crop and downscale CheXpert to 1024x1024
├── eval.py                      # Metrics, bootstrap confidence intervals, p-values, ROC curves
├── tableCreator.py              # Tables 1-3 of the paper
├── supplementTables.py          # Tables of the Supplementary Information
├── plotThresholds.py            # Optimal threshold plots
├── benchmarkLatency.py          # Latency benchmark
├── benchmarkPrompts.json        # Prompts for the latency benchmark
├── utilFunc.py                  # General utility functions
├── splits/                      # Images of the validation and test splits (for reference)
├── output/                      # Model outputs of all inference runs
├── evalOutput/                  # Evaluation results
├── benchmarkOut/                # Latency measurements
└── results_tables.txt           # Reference output of tableCreator.py
```

## Citation

If you find this code helpful for your research, please cite our paper:

Wienholt, P., Hauke, F., Kuhl, C., Kather, J. N., Nebelung, S., Truhn, D. Single-Pass Decision Scores Enable Rapid Classification with Adjustable Thresholds in Medical Multimodal Large Language Models. (2026)

## References
[1] Irvin, J., Rajpurkar, P., Ko, M. et al. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. Proceedings of the AAAI Conference on Artificial Intelligence 33, 590–597 (2019). https://doi.org/10.1609/aaai.v33i01.3301590

[2] Saporta, A., Gui, X., Agrawal, A. et al. Benchmarking saliency methods for chest X-ray interpretation. Nat Mach Intell 4, 867–878 (2022). https://doi.org/10.1038/s42256-022-00536-x

[3] Johnson, A., Lungren, M., Peng, Y. et al. MIMIC-CXR-JPG - chest radiographs with structured labels (version 2.0.0). PhysioNet (2019). https://doi.org/10.13026/8360-t248

[4] Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161

[5] Yilmaz, A., Yasar, S. P., Gencoglan, G. & Temelkuran, B. DERM12345: A Large, Multisource Dermatoscopic Skin Lesion Dataset with 40 Subclasses. Sci Data 11, 1302 (2024). https://doi.org/10.1038/s41597-024-04104-3

[6] Kather, J. N., Halama, N. & Marx, A. 100,000 histological images of human colorectal cancer and healthy tissue. Zenodo (2018). https://doi.org/10.5281/zenodo.1214456

[7] Bajwa, M. N., Singh, G. A. P., Neumeier, W. et al. G1020: A Benchmark Retinal Fundus Image Dataset for Computer-Aided Glaucoma Detection. 2020 International Joint Conference on Neural Networks (IJCNN), 1–7 (2020). https://doi.org/10.1109/IJCNN48605.2020.9207664
