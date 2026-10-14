"""Latency benchmark of our approach: one forward pass, the model answers yes/no with a single token,
and the probability is computed from the yes/no log-probabilities.

usage:
  python3 benchmarkLatency.py export   # locally: collect the prompts of all datasets from the inference pickles
  python3 benchmarkLatency.py <MedGemma4B|MedGemma27B|Gemma4B|Gemma27B>   # on a GPU node

Prompts come from all datasets; images are randomly sampled from several datasets. Every query
uses a fresh image so that vLLM's prefix cache cannot reuse image tokens between queries.
"""
import sys
import os
import json
import glob
import time
import pickle
import random
import numpy as np
from PIL import Image

PROMPT_FILE = "benchmarkPrompts.json"
PICKLE_FOLDER = "output/"
PROMPT_DATASET_LIST = ["CheXpert", "MIMIC", "HAM10000", "DERM12345", "NCT_CRC_HE_100K_1000", "G1020"]
IMAGE_GLOB_LIST = [
    "data/HAM10000/*.jpg",
    "data/DERM12345/**/*.jpg",
    "data/G1020/**/*.jpg",
    "data/NCT-CRC-HE-100K/*/*.tif",
    "data/CheXpert/test/**/*.jpg",
]
IMAGES_PER_FOLDER = 150
PROMPTS_PER_DATASET = 10  # single-query measurement: labels sampled per dataset
IMAGES_PER_DATASET = 5    # all-labels-per-image measurement
BATCHES_PER_DATASET = 3   # throughput measurement, batches of BATCH_IMAGES images x all labels
BATCH_IMAGES = 4
N_WARMUP = 5
SEED = 0
OUT_FOLDER = "benchmarkOut/"


def exportPrompts():
    prompts = {}
    for dataset in PROMPT_DATASET_LIST:
        with open(PICKLE_FOLDER + f"eval_MedGemma27B_{dataset}.pkl", "rb") as f:
            dataDict = pickle.load(f)
        prompts[dataset] = dataDict["promptList"]
    with open(PROMPT_FILE, "w") as f:
        json.dump(prompts, f, indent=1)
    print({k: len(v) for k, v in prompts.items()})


def sampleImagePaths(rng):
    paths = []
    for pattern in IMAGE_GLOB_LIST:
        folderPaths = sorted(glob.glob(pattern, recursive=True))
        assert folderPaths, f"no images for {pattern}"
        paths += rng.sample(folderPaths, IMAGES_PER_FOLDER)
    rng.shuffle(paths)
    return paths


def summarize(durations):
    q25, q50, q75 = np.percentile(durations, [25, 50, 75])
    return {"median_s": q50, "q25_s": q25, "q75_s": q75, "n": len(durations)}


def timed(fn, *args):
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def runBenchmark(modelName):
    from runLLM import getModel
    model = getModel(modelName)
    rng = random.Random(SEED)
    with open(PROMPT_FILE) as f:
        prompts = json.load(f)
    imgIter = (Image.open(p).convert("RGB") for p in sampleImagePaths(rng))

    yesNoProb = lambda ps, imgs: model.getRawLogProbs(ps, [[x] for x in imgs], ["no", "yes"])
    for _ in range(N_WARMUP):
        yesNoProb(prompts["CheXpert"][:1], [next(imgIter)])

    results = {"model": modelName, "seed": SEED, "perDataset": {}}
    allSingle, allPerImage = [], []
    for dataset in PROMPT_DATASET_LIST:
        datasetPrompts = prompts[dataset]
        single = [timed(yesNoProb, [p], [next(imgIter)])
                  for p in rng.sample(datasetPrompts, min(PROMPTS_PER_DATASET, len(datasetPrompts)))]
        perImage = [timed(yesNoProb, datasetPrompts, [img] * len(datasetPrompts))
                    for img in (next(imgIter) for _ in range(IMAGES_PER_DATASET))]
        batchDurations = []
        for _ in range(BATCHES_PER_DATASET):
            imgs = [next(imgIter) for _ in range(BATCH_IMAGES)]
            batchDurations.append(timed(yesNoProb, datasetPrompts * BATCH_IMAGES, [img for img in imgs for _ in datasetPrompts]))
        results["perDataset"][dataset] = {
            "labels": len(datasetPrompts),
            "singleQuery": summarize(single),
            "allLabelsOneImage": summarize(perImage),
            "throughput_queries_per_s": BATCHES_PER_DATASET * BATCH_IMAGES * len(datasetPrompts) / sum(batchDurations),
        }
        allSingle += single
        allPerImage += [d / len(datasetPrompts) for d in perImage]
        print(dataset, results["perDataset"][dataset], flush=True)

    results["overall"] = {"singleQuery": summarize(allSingle), "perLabelWhenAllLabelsOneImage": summarize(allPerImage)}
    print(json.dumps(results["overall"], indent=2))
    os.makedirs(OUT_FOLDER, exist_ok=True)
    with open(OUT_FOLDER + f"latency_{modelName}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    if sys.argv[1] == "export":
        exportPrompts()
    else:
        runBenchmark(sys.argv[1])
