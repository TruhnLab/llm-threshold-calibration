from torch.utils.data import Dataset
from utilFunc import readCSV
import random
from PIL import Image
from functools import lru_cache

HAM10000_BASE_PATH = "data/HAM10000/"

DIAG_1_LABEL_LIST = ['Benign','Indeterminate', 'Malignant']
DIAG_2_LABEL_LIST = ['Benign epidermal proliferations', 'Benign melanocytic proliferations', 'Benign soft tissue proliferations - Fibro-histiocytic', 'Benign soft tissue proliferations - Vascular', 'Indeterminate epidermal proliferations', 'Malignant adnexal epithelial proliferations - Follicular', 'Malignant epidermal proliferations', 'Malignant melanocytic proliferations (Melanoma)']
DIAG_3_LABEL_LIST = ['Basal cell carcinoma', 'Dermatofibroma', 'Melanoma, NOS', 'Nevus', 'Pigmented benign keratosis', 'Solar or actinic keratosis', 'Squamous cell carcinoma, NOS']
LABEL_LIST = DIAG_1_LABEL_LIST + DIAG_2_LABEL_LIST + DIAG_3_LABEL_LIST

LABEL_DESCRIPTIONS = {
    'Benign': 'a benign skin lesion',
    'Indeterminate': 'a skin lesion with an indeterminate diagnosis',
    'Malignant': 'a malignant skin lesion',
    'Benign epidermal proliferations': 'benign epidermal proliferation',
    'Benign melanocytic proliferations': 'benign melanocytic proliferation',
    'Benign soft tissue proliferations - Fibro-histiocytic': 'benign fibro-histiocytic proliferation',
    'Benign soft tissue proliferations - Vascular': 'benign vascular proliferation',
    'Indeterminate epidermal proliferations': 'indeterminate epidermal proliferation requiring further evaluation',
    'Malignant adnexal epithelial proliferations - Follicular': 'malignant follicular adnexal epithelial proliferation',
    'Malignant epidermal proliferations': 'malignant epidermal proliferation',
    'Malignant melanocytic proliferations (Melanoma)': 'malignant melanocytic proliferation (melanoma)',
    'Basal cell carcinoma': 'basal cell carcinoma',
    'Dermatofibroma': 'dermatofibroma',
    'Melanoma, NOS': 'melanoma, not otherwise specified',
    'Nevus': 'a nevus (melanocytic nevus)',
    'Pigmented benign keratosis': 'pigmented benign keratosis (seborrheic keratosis)',
    'Solar or actinic keratosis': 'solar or actinic keratosis',
    'Squamous cell carcinoma, NOS': 'squamous cell carcinoma, not otherwise specified'
}

HAM10000_PROMPT = """You are a medical image analysis assistant.

TASK
Based on the visual evidence in the image, answer the following question with "yes" or "no".

Question: Does the lesion in the image show features of {finding}?
"""

HAM10000_PROB_PROMPT = """You are a medical image analysis assistant.

TASK
Based on the visual evidence in the image, estimate the probability that the following finding is present.
Provide the probability as a JSON object with a single key "probability" and a value between 0.000 and 1.000.

Finding: {finding}
"""

HAM10000_JSON_TEMPLATE = {
    "type": "object",
    "properties": {
        "probability": {
            "type": "string",
            "pattern": r"[01]\.\d{3}"
        }
    },
    "required": ["probability"]
}

def squareCrop(img):
    width, height = img.size
    if width == height: return img
    newSize = min(width,height)
    left = (width - newSize)/2
    top = (height - newSize)/2
    right = (width + newSize)/2
    bottom = (height + newSize)/2
    img = img.crop((left, top, right, bottom))
    return img

class HAM10000(Dataset):
    def __init__(self,splitEnd,splitStart=0,basePath = HAM10000_BASE_PATH,shuffle=True):
        self.basePath = basePath
        metaFile = basePath + "ham10000_metadata_2025-07-18.csv"
        self.csvHead,*self.lineList = readCSV(metaFile)
        if shuffle: random.Random(0).shuffle(self.lineList)
        self.lineList = self.lineList[splitStart:splitEnd]

    def __len__(self):
        return len(self.lineList)

    @lru_cache
    def getLabelNames(self):
        return [LABEL_DESCRIPTIONS[x] for x in LABEL_LIST]

    @lru_cache
    def getPromptList(self):
        return [HAM10000_PROMPT.replace("{finding}",x) for x in self.getLabelNames()]

    @lru_cache
    def getProbPromptList(self):
        return [(HAM10000_PROB_PROMPT.replace("{finding}",x), HAM10000_JSON_TEMPLATE) for x in self.getLabelNames()]

    def __getitem__(self, index):
        line = self.lineList[index]
        label = [0]*len(LABEL_LIST)
        for diagnosis in line[7:10]: # diagnosis 1-3; empty = no finer diagnosis, sets no label
            if diagnosis: label[LABEL_LIST.index(diagnosis)] = 1
        imgPath = self.basePath + line[0] + ".jpg"
        img = Image.open(imgPath)
        img = squareCrop(img).convert("RGB")
        return [img],label
