from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
import random
from functools import lru_cache

NCT_CRC_HE_100K_DATASET_PATH = "data/NCT-CRC-HE-100K/"

NCT_CRC_HE_100K_PROMPT = """You are a digital pathology assistant.

TASK
Based on the visual evidence in the image, answer the following question with "yes" or "no".

Question: Is the dominant tissue type in this image patch {finding}?"""

NCT_CRC_HE_100K_PROB_PROMPT = """You are a digital pathology assistant.

TASK
Based on the visual evidence in the image, estimate the probability that the following finding is present.
Provide the probability as a JSON object with a single key "probability" and a value between 0.000 and 1.000.

Finding: Is the dominant tissue type in this image patch {finding}?
"""

NCT_CRC_HE_100K_JSON_TEMPLATE = {
    "type": "object",
    "properties": {
        "probability": {
            "type": "string",
            "pattern": r"[01]\.\d{3}"
        }
    },
    "required": ["probability"]
}

labelDict = {
    #from https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/NCT-CRC-HE-100K.md
    "ADI":"adipose tissue",
    "BACK":"background elements",
    "DEB":"debris",
    "LYM":"lymphocytes",
    "MUC":"mucus",
    "MUS":"smooth muscle",
    "NORM":"normal colon mucosa",
    "STR":"cancer-associated stroma",
    "TUM":"colorectal adenocarcinoma epithelial cells",
}


class NCT_CRC_HE_100K(Dataset):
    def __init__(self,split,datasetPath=NCT_CRC_HE_100K_DATASET_PATH,shuffle=True,splitStart=0):
        self.data = ImageFolder(datasetPath)
        self.classList = [x for x in self.data.classes if x in labelDict]
        totalCount = len(self.data)
        idList = list(range(totalCount))
        if shuffle: random.Random(0).shuffle(idList) #deterministic shuffle => same for 2 dataset splits

        if type(split) == int:
            self.idxList = idList[splitStart:split]
        else:
            trainValidCutoff = int(totalCount*0.6)
            validTestCutoff = int(totalCount*0.8)
            trainList = idList[:trainValidCutoff]
            validList = idList[trainValidCutoff:validTestCutoff]
            testList = idList[validTestCutoff:]
            self.idxList = {"train":trainList,"valid":validList,"test":testList}[split]
        # keep only the 9 tissue classes (in case additional folders are present)
        self.idxList = [i for i in self.idxList if self.data.classes[self.data.targets[i]] in labelDict]
    
    @lru_cache
    def getLabelNames(self):
        return [labelDict[x].lower() for x in self.classList]

    @lru_cache
    def getPromptList(self):
        return [NCT_CRC_HE_100K_PROMPT.replace("{finding}",x) for x in self.getLabelNames()]

    @lru_cache
    def getProbPromptList(self):
        return [(NCT_CRC_HE_100K_PROB_PROMPT.replace("{finding}",x), NCT_CRC_HE_100K_JSON_TEMPLATE) for x in self.getLabelNames()]

    
    def __len__(self):
        return len(self.idxList)

    def __getitem__(self, idx):
        globalIdx = self.idxList[idx]
        img, labelIdx = self.data[globalIdx]
        label = [0]*len(self.classList)
        label[self.classList.index(self.data.classes[labelIdx])] = 1
        return [img],label




