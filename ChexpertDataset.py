from torch.utils.data import Dataset
from utilFunc import readCSV
from torchvision.transforms import v2
from PIL import Image
import torchvision.transforms.functional
import torch
from functools import lru_cache

TEST = "test"
TRAIN = "train"
VALIDATE = "validate"

IMG_SIZE = 1024

POSSIBLE_SPLITS = [
    TEST,
    TRAIN,
    VALIDATE,
]



CHEXPERT_BASE_PATH = "data/CheXpert/" # preprocessed to 1024x1024 with preprocessCheXpert.py

META_DICT = {
    TRAIN:"train_visualCheXbert.csv",
    VALIDATE:"valid.csv",
    TEST:"test_labels.csv",
}

META_LABEL_CUT = 5
META_LABEL_CUT_DICT = {
    TRAIN:5,
    VALIDATE:5,
    TEST:1,
}

CHEXPERT_PROMPT = """You are a medical image analysis assistant.

You will be provided with medical imaging and one clinical statement.  

TASK  
Decide whether the imaging evidence supports the clinical statement **exactly as written**.

Clinical statement: The imaging shows {finding}.

If the imaging evidence supports the statement, answer with: yes  
If the imaging evidence contradicts the statement or is indeterminate, answer with: no"""

CHEXPERT_PROB_PROMPT = """You are a medical image analysis assistant.

TASK
Based on the visual evidence in the image, estimate the probability that the following finding is present.
Provide the probability as a JSON object with a single key "probability" and a value between 0.000 and 1.000.

Finding: {finding}
"""

CHEXPERT_JSON_TEMPLATE = {
    "type": "object",
    "properties": {
        "probability": {
            "type": "string",
            "pattern": r"[01]\.\d{3}"
        }
    },
    "required": ["probability"]
}


def shiftVisualChexpert(csvCont): # align with normal chexpert training data
    labelCut = 5
    for line in csvCont:
        lastEl = line.pop()
        line.insert(labelCut,lastEl)
    print("SHIFTED VISUAL CHEXPERT TRAINING FILE")


class ClassificationChexpertDataset(Dataset):
    def __init__(self, usedSplit, augementImg=False, imgSize=IMG_SIZE):
        self.usedSplit = usedSplit
        self.augementImg = augementImg
        self.imgSize = imgSize
        metaFile = CHEXPERT_BASE_PATH + META_DICT[self.usedSplit]
        csvCont = readCSV(metaFile)
        if META_DICT[self.usedSplit] == "train_visualCheXbert.csv": shiftVisualChexpert(csvCont)#shift dataset
        head,*data = csvCont

        labelCut = META_LABEL_CUT_DICT[usedSplit]
        self.labelNames = head[labelCut:]

        getTotalPath = lambda relativePath: CHEXPERT_BASE_PATH + relativePath.replace("CheXpert-v1.0/","")

        labelDict = { #set unknown to 0
            "": 0,
            "1.0": 1,
            "0.0": 0,
            "-1.0": 0,
        }
        labelToInt = lambda lst: [labelDict[x] for x in lst]
        self.itemList = [(getTotalPath(x[0]),labelToInt(x[labelCut:])) for x in data]
    
    @lru_cache
    def getLabelNames(self):
        return [x.lower() for x in self.labelNames]
    
    @lru_cache
    def getPromptList(self):
        return [CHEXPERT_PROMPT.replace("{finding}",x) for x in self.getLabelNames()]

    @lru_cache
    def getProbPromptList(self):
        return [(CHEXPERT_PROB_PROMPT.replace("{finding}",x), CHEXPERT_JSON_TEMPLATE) for x in self.getLabelNames()]

    
    def __len__(self):
        return len(self.itemList)
    
    def __getitem__(self, idx):
        imgPath,label = self.itemList[idx]
        img = self.getImg(imgPath)

        return [img],label

    def getImg(self,imgPath):
        img = Image.open(imgPath).convert("RGB")
        img = img.resize((IMG_SIZE,IMG_SIZE))
        return img