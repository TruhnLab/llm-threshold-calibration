from VLLMModel import VLLMModel, makeBatch

from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from functools import lru_cache
import os
import sys

MODEL_LIST = ["MedGemma4B","MedGemma27B","Gemma4B","Gemma27B"]
MODEL_TO_USE = sys.argv[1] # usage: python3 runLLM.py <MedGemma4B|MedGemma27B|Gemma4B|Gemma27B>
assert MODEL_TO_USE in MODEL_LIST

BATCH_SIZE = 4

OUT_FOLDER = "output/"

DATASET_LIST = ["CheXpert","G1020","DERM12345","HAM10000","NCT_CRC_HE_100K_1000", "MIMIC"]

@lru_cache(maxsize=1)
def getModel(usedModel):
    if usedModel == "MedGemma4B":
        model = VLLMModel(modelName="google/medgemma-4b-it",revision="1c73d61525d5e0d32f0f5e9b8aadf33b55d6676c")
    if usedModel == "MedGemma27B":
        model = VLLMModel(modelName="google/medgemma-27b-it",revision="2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f")
    if usedModel == "Gemma4B":
        model = VLLMModel(modelName="google/gemma-3-4b-it",revision="093f9f388b31de276ce2de164bdc2081324b9767")
    if usedModel == "Gemma27B":
        model = VLLMModel(modelName="google/gemma-3-27b-it",revision="005ad3404e59d6023443cb575daa05336842228a")
    return model

def getDataset(usedDataset,split):
    assert split in ["test","valid"]
    if usedDataset == "CheXpert":
        from ChexpertDataset import ClassificationChexpertDataset
        if split == "test":
            dataset = ClassificationChexpertDataset("test")
        if split == "valid":
            dataset = ClassificationChexpertDataset("validate")

    if usedDataset == "NCT_CRC_HE_100K_1000":
        from NCT_CRC_HE_100K import NCT_CRC_HE_100K
        if split == "test":
            dataset = NCT_CRC_HE_100K(split=5000)
        if split == "valid":
            dataset = NCT_CRC_HE_100K(split=10000,splitStart=5000)
        assert len(dataset) == 5000

    if usedDataset == "MIMIC":
        from MIMIC_Dataset import MIMIC_Dataset
        if split == "test":
            dataset = MIMIC_Dataset("test")
        if split == "valid":
            dataset = MIMIC_Dataset("validate")
    
    if usedDataset == "G1020":
        from G1020 import G1020
        dataset = G1020(split)

    
    if usedDataset == "HAM10000":
        from HAM10000 import HAM10000
        if split == "test":
            dataset = HAM10000(splitEnd=5000)
        if split == "valid":
            dataset = HAM10000(splitEnd=10000,splitStart=5000)
        assert len(dataset) == 5000

    
    if usedDataset == "DERM12345":
        from DERM12345 import DERM12345
        if split == "test":
            dataset = DERM12345(splitEnd=5000)
        if split == "valid":
            dataset = DERM12345(splitEnd=10000,splitStart=5000)
        assert len(dataset) == 5000
    
    print("LOADED",usedDataset,split,"len:",len(dataset))
    dataLoader = DataLoader(dataset,batch_size=BATCH_SIZE,collate_fn=lambda x:x,num_workers=4)
    return dataset,dataLoader

def sampleToPromptImgList(sample,promptList,textProbInfereList):
    imgList, labelList = sample
    jsonPromptList = [x[0] for x in textProbInfereList]
    return promptList,[imgList]*len(promptList),labelList,jsonPromptList

def runBatchEval(dataLoader, dataset, model, usedDataset):
    allJsonProbList = []
    allLogProbs = []
    allTargetList = []
    print("USED LABEL:", dataset.getLabelNames())
    print("USED MODEL:", MODEL_TO_USE)
    print("USED DATASET", usedDataset)

    for batch in tqdm(dataLoader):
        batchElements = [sampleToPromptImgList(x,dataset.getPromptList(),dataset.getProbPromptList()) for x in batch]
        jsonTemplate = dataset.getProbPromptList()[0][1]

        promptList = []
        imgList = []
        labelList = []
        jsonPromptList = []
        for sample in batchElements:
            promptList.extend(sample[0])
            imgList.extend(sample[1])
            labelList.append(sample[2])
            jsonPromptList.extend(sample[3])

        jsonProbs = model(jsonPromptList,imgList,jsonTemplate=jsonTemplate,max_tokens=64,top_k=1)
        jsonProbs = makeBatch(jsonProbs, len(dataset.getLabelNames()))
        logProbs = model.getRawLogProbs(promptList, imgList, ["no", "yes"])

        outPred = makeBatch(logProbs, len(dataset.getLabelNames()))
        assert len(outPred[0]) == len(dataset.getLabelNames())
        allLogProbs .extend(outPred)
        allTargetList.extend(labelList)
        allJsonProbList.extend(jsonProbs)

    return allLogProbs , allTargetList, allJsonProbList

def execEval(usedModel,usedDataset):
    testDataset,testDataLoader = getDataset(usedDataset,"test")
    validDataset,validDataLoader = getDataset(usedDataset,"valid")
    model = getModel(usedModel)

    print("VALID:")
    validLogProbs, validAllTargetList, validAllJsonProbList = runBatchEval(validDataLoader, validDataset, model, usedDataset)
    print("TEST:")
    testLogProbs, testAllTargetList, testAllJsonProbList = runBatchEval(testDataLoader, testDataset, model, usedDataset)

    saveDict = {
        "testLogProbs":testLogProbs,
        "testTargets":testAllTargetList,
        "testJsonProbList":testAllJsonProbList,
        "validLogProbs":validLogProbs,
        "validTargets":validAllTargetList,
        "validJsonProbList":validAllJsonProbList,
        "labelNames":testDataset.getLabelNames(),
        "testDatasetName":usedDataset,
        "modelName":usedModel,
        "promptList":testDataset.getPromptList(),
        "jsonPrompts":testDataset.getProbPromptList(),
        }

    os.makedirs(OUT_FOLDER,exist_ok=True)
    pklName = OUT_FOLDER + "eval_" + usedModel + "_" + usedDataset + ".pkl"
    with open(pklName, "wb") as f:
        pickle.dump(saveDict, f)
    print("DONE")



if __name__ == "__main__":
    for usedDataset in DATASET_LIST:
        execEval(MODEL_TO_USE,usedDataset)
