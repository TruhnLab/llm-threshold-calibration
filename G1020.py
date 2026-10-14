from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import random
from functools import lru_cache

G1020_BASE_PATH = "data/G1020/"



labelDict = {
    "normal": "a normal eye",
    "glaucoma": "glaucoma",
}

G1020_PROMPT = """You are a retinal image analysis assistant.

TASK
Based on the visual evidence in the image, answer the following question with "yes" or "no".

Question: Does the image show signs of {finding}?
"""

G1020_PROB_PROMPT = """You are a retinal image analysis assistant.

TASK
Based on the visual evidence in the image, estimate the probability that the following finding is present.
Provide the probability as a JSON object with a single key "probability" and a value between 0.000 and 1.000.

Finding: {finding}
"""

G1020_JSON_TEMPLATE = {
    "type": "object",
    "properties": {
        "probability": {
            "type": "string",
            "pattern": r"[01]\.\d{3}"
        }
    },
    "required": ["probability"]
}

class G1020(Dataset):
    def __init__(self,split,datasetPath=G1020_BASE_PATH,shuffle=True,datasetCut=None):
        if split == "test":
            usedPath = datasetPath + "Testing"
        elif split == "valid":
            usedPath = datasetPath + "Training"
        self.data = ImageFolder(usedPath)
        self.classList = self.data.classes
        print(self.classList)
        totalCount = len(self.data)
        self.idxList = list(range(totalCount))
        if shuffle: random.Random(0).shuffle(self.idxList)
        if datasetCut is not None:
            self.idxList = self.idxList[:datasetCut]
        
    @lru_cache
    def getLabelNames(self):
        return [labelDict[x].lower() for x in self.classList]

    @lru_cache
    def getPromptList(self):
        return [G1020_PROMPT.replace("{finding}",x) for x in self.getLabelNames()]

    @lru_cache
    def getProbPromptList(self):
        return [(G1020_PROB_PROMPT.replace("{finding}",x), G1020_JSON_TEMPLATE) for x in self.getLabelNames()]

    
    def __len__(self):
        return len(self.idxList)

    def __getitem__(self, idx):
        globalIdx = self.idxList[idx]
        img, labelIdx = self.data[globalIdx]
        label = [0]*len(self.classList)
        label[labelIdx] = 1
        return [img],label