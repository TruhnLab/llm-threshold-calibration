from torch.utils.data import Dataset
from utilFunc import readCSV
import random
from PIL import Image
from functools import lru_cache

DIAG_1_LABEL_LIST = ['Benign', 'Indeterminate', 'Malignant']
DIAG_2_LABEL_LIST = ['Benign epidermal proliferations', 'Benign melanocytic proliferations', 'Benign soft tissue proliferations - Fibro-histiocytic', 'Benign soft tissue proliferations - Vascular', 'Flat melanotic pigmentations - not melanocytic nevus', 'Indeterminate epidermal proliferations', 'Malignant adnexal epithelial proliferations - Apocrine or Eccrine', 'Malignant adnexal epithelial proliferations - Follicular', 'Malignant epidermal proliferations', 'Malignant melanocytic proliferations (Melanoma)', 'Malignant soft tissue proliferations - Fibro-histiocytic', 'Malignant soft tissue proliferations - Vascular']
DIAG_3_LABEL_LIST = ['Angiokeratoma', 'Basal cell carcinoma', 'Dermatofibroma', 'Dermatofibrosarcoma protuberans', 'Hemangioma', 'Ink-spot lentigo', 'Kaposi sarcoma', 'Lentigo simplex', 'Lichen planus like keratosis', 'Lymphangioma', 'Melanoma Invasive', 'Melanoma in situ', 'Melanoma, NOS', 'Nevus', 'Paget disease', 'Pyogenic granuloma', 'Seborrheic keratosis', 'Solar lentigo', 'Solar or actinic keratosis', 'Squamous cell carcinoma in situ', 'Squamous cell carcinoma, NOS', 'Vascular spider']
LABEL_LIST = DIAG_1_LABEL_LIST + DIAG_2_LABEL_LIST + DIAG_3_LABEL_LIST



LABEL_DESCRIPTIONS = {
    # DIAG_1
    'Benign': 'a benign skin lesion',
    'Indeterminate': 'a skin lesion with an indeterminate diagnosis',
    'Malignant': 'a malignant skin lesion',

    # DIAG_2
    'Benign epidermal proliferations': 'benign epidermal proliferation',
    'Benign melanocytic proliferations': 'benign melanocytic proliferation',
    'Benign soft tissue proliferations - Fibro-histiocytic': 'benign fibro-histiocytic soft tissue proliferation',
    'Benign soft tissue proliferations - Vascular': 'benign vascular soft tissue proliferation',
    'Flat melanotic pigmentations - not melanocytic nevus': 'flat melanotic pigmentation (not a melanocytic nevus)',
    'Indeterminate epidermal proliferations': 'indeterminate epidermal proliferation',
    'Malignant adnexal epithelial proliferations - Apocrine or Eccrine': 'malignant apocrine or eccrine adnexal epithelial proliferation',
    'Malignant adnexal epithelial proliferations - Follicular': 'malignant follicular adnexal epithelial proliferation',
    'Malignant epidermal proliferations': 'malignant epidermal proliferation',
    'Malignant melanocytic proliferations (Melanoma)': 'malignant melanocytic proliferation (melanoma)',
    'Malignant soft tissue proliferations - Fibro-histiocytic': 'malignant fibro-histiocytic soft tissue proliferation',
    'Malignant soft tissue proliferations - Vascular': 'malignant vascular soft tissue proliferation',

    # DIAG_3
    'Angiokeratoma': 'angiokeratoma',
    'Basal cell carcinoma': 'basal cell carcinoma',
    'Dermatofibroma': 'dermatofibroma',
    'Dermatofibrosarcoma protuberans': 'dermatofibrosarcoma protuberans',
    'Hemangioma': 'hemangioma',
    'Ink-spot lentigo': 'ink-spot lentigo',
    'Kaposi sarcoma': 'Kaposi sarcoma',
    'Lentigo simplex': 'lentigo simplex',
    'Lichen planus like keratosis': 'lichen planus like keratosis',
    'Lymphangioma': 'lymphangioma',
    'Melanoma Invasive': 'invasive melanoma',
    'Melanoma in situ': 'melanoma in situ',
    'Melanoma, NOS': 'melanoma, not otherwise specified',
    'Nevus': 'nevus',
    'Paget disease': 'Paget disease',
    'Pyogenic granuloma': 'pyogenic granuloma',
    'Seborrheic keratosis': 'seborrheic keratosis',
    'Solar lentigo': 'solar lentigo',
    'Solar or actinic keratosis': 'solar or actinic keratosis',
    'Squamous cell carcinoma in situ': 'squamous cell carcinoma in situ',
    'Squamous cell carcinoma, NOS': 'squamous cell carcinoma, not otherwise specified',
    'Vascular spider': 'vascular spider'
}

DERM12345_PROMPT = """You are a medical image analysis assistant.

TASK
Based on the visual evidence in the image, answer the following question with "yes" or "no".

Question: Does the lesion in the image show features of {finding}?
"""

DERM12345_PROB_PROMPT = """You are a medical image analysis assistant.

TASK
Based on the visual evidence in the image, estimate the probability that the following finding is present.
Provide the probability as a JSON object with a single key "probability" and a value between 0.000 and 1.000.

Finding: {finding}
"""

DERM12345_JSON_TEMPLATE = {
    "type": "object",
    "properties": {
        "probability": {
            "type": "string",
            "pattern": r"[01]\.\d{3}"
        }
    },
    "required": ["probability"]
}

DERM12345_BASE_PATH = "data/DERM12345/"

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

class DERM12345(Dataset):
    def __init__(self,splitEnd,splitStart=0,basePath = DERM12345_BASE_PATH,shuffle=True):
        self.basePath = basePath
        metaFile = basePath + "metadata.csv"
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
        return [DERM12345_PROMPT.replace("{finding}",x) for x in self.getLabelNames()]
    
    @lru_cache
    def getProbPromptList(self):
        return [(DERM12345_PROB_PROMPT.replace("{finding}",x),DERM12345_JSON_TEMPLATE) for x in self.getLabelNames()]

    def __getitem__(self, index):
        line = self.lineList[index]
        label = [0]*len(LABEL_LIST)
        for diagnosis in line[5:8]: # diagnosis 1-3; empty = no finer diagnosis, sets no label
            if diagnosis: label[LABEL_LIST.index(diagnosis)] = 1
        imgPath = self.basePath + line[0] + ".jpg"
        img = Image.open(imgPath)
        img = squareCrop(img).convert("RGB")
        return [img],label
