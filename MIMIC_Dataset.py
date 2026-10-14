from torch.utils.data import Dataset
from utilFunc import readCSV
import os
from PIL import Image
from functools import lru_cache


MIMIC_CXR_PATH = "data/MIMIC-CXR-JPG/"

POSSIBLE_SPLITS = ["train","test","validate"]

def squareCrop(img):
    width, height = img.size   # Get dimensions
    if width == height: return img
    newSize = min(width,height)

    left = (width - newSize)/2
    top = (height - newSize)/2
    right = (width + newSize)/2
    bottom = (height + newSize)/2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img

MIMIC_PROMPT = """You are a medical image analysis assistant.

You will be provided with medical imaging and one clinical statement.  

TASK  
Decide whether the imaging evidence supports the clinical statement **exactly as written**.

Clinical statement: The imaging shows {finding}.

If the imaging evidence supports the statement, answer with: yes  
If the imaging evidence contradicts the statement or is indeterminate, answer with: no"""

MIMIC_PROB_PROMPT = """You are a medical image analysis assistant.

TASK
Based on the visual evidence in the image, estimate the probability that the following finding is present.
Provide the probability as a JSON object with a single key "probability" and a value between 0.000 and 1.000.

Finding: {finding}
"""

MIMIC_JSON_TEMPLATE = {
    "type": "object",
    "properties": {
        "probability": {
            "type": "string",
            "pattern": r"[01]\.\d{3}"
        }
    },
    "required": ["probability"]
}

class MIMIC_Dataset(Dataset):
    def __init__(self,split,cxrBasePath=MIMIC_CXR_PATH):
        assert split in POSSIBLE_SPLITS
        self.cxrBasePath = cxrBasePath
        self._load_split_data(split, cxrBasePath)
        self._load_chexpert_labels(cxrBasePath)
    
    def _iterateStudyIds(self):
        self.subjectStudyList = list()
        for subjId in list(self.subjectDict.keys()):
            for studyId in self.subjectDict[subjId].keys():
                self.subjectStudyList.append((subjId,studyId))
        


    def _addCXRTime(self):
        metaHead,*metaData = readCSV(self.cxrBasePath+"mimic-cxr-2.0.0-metadata.csv")
        for line in metaData:
            lineDict = dict(zip(metaHead,line))
            if lineDict["subject_id"] not in self.subjectDict: continue
            studyTime = ("0"*6+str(int(float(lineDict["StudyTime"]))))[-6:]
            studyTime = studyTime[0:2] + ":" + studyTime[2:4] + ":" + studyTime[4:6]
            studyDate = lineDict["StudyDate"]
            studyDate = studyDate[0:4] + "-" + studyDate[4:6] + "-" + studyDate[6:8]
            timeStr = studyDate + " " + studyTime
            self.subjectDict[lineDict["subject_id"]][lineDict["study_id"]]["time"] = timeStr


    def _load_split_data(self, split, cxrBasePath):
        splitFileHead, *splitFileData = readCSV(cxrBasePath + "mimic-cxr-2.0.0-split.csv")
        
        self.subjectDict = dict()
        for line in splitFileData:
            if line[3] != split: continue
            subjID = line[2]
            studyID = line[1]
            if subjID not in self.subjectDict: self.subjectDict[subjID] = dict()
            self.subjectDict[subjID][studyID] = {"dicom_id":line[0],"study_id":line[1],"subject_id":line[2],"split":line[3]}
        self._iterateStudyIds()
        self._addCXRTime()

    def _load_chexpert_labels(self, cxrBasePath):
        chexpertLabelHead,*chexpertLabelData = readCSV(cxrBasePath + "mimic-cxr-2.0.0-chexpert.csv")
        self.labelNames = chexpertLabelHead[2:]
        labelMap = {"":0,"1.0":1,"-1.0":0,"0.0":0}
        for line in chexpertLabelData:
            subjID = line[0]
            studyID = line[1]
            if subjID not in self.subjectDict: continue
            labelList = [labelMap[x] for x in line[2:]]
            self.subjectDict[subjID][studyID]["label"] = labelList
            self.subjectDict[subjID][studyID]["labelNames"] = self.labelNames.copy()
        
    def _getImgList(self,subjId,studyId):
        path = self.cxrBasePath +"files/"+ "p" + subjId[:2] + "/p" + subjId + "/s" + studyId + "/"
        imgPathList = [path + x for x in os.listdir(path)]
        imgList = [Image.open(x).convert("RGB") for x in imgPathList]
        imgList = [squareCrop(x) for x in imgList]
        return imgList
    
    @lru_cache
    def getLabelNames(self):
        return [x.lower() for x in self.labelNames]

    @lru_cache
    def getPromptList(self):
        return [MIMIC_PROMPT.replace("{finding}",x) for x in self.getLabelNames()]

    @lru_cache
    def getProbPromptList(self):
        return [(MIMIC_PROB_PROMPT.replace("{finding}",x), MIMIC_JSON_TEMPLATE) for x in self.getLabelNames()]

    def __len__(self):
        return len(self.subjectStudyList)

    def __getitem__(self, idx):
        subjId,studyId = self.subjectStudyList[idx]
        imgList = self._getImgList(subjId,studyId)

        label = self.subjectDict[subjId][studyId]["label"]
        return imgList,label