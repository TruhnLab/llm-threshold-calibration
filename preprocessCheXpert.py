"""Center-crop the CheXpert validation and test images (from CheXlocalize) to a square and downscale them to 1024x1024 (as used in the paper).
Adapted from https://github.com/TruhnLab/MedicalPatchNet/blob/main/preprocessing/preProcCheXpert.py"""
import csv
from PIL import Image
import os
import shutil
import multiprocessing as mp

CROP_SIZE = 1024

CHEXLOCALIZE_FOLDER = "data/chexlocalize/CheXpert/" # CheXlocalize download: CheXpert validation set (val/, val_labels.csv) and test set (test/, test_labels.csv)
OUT_FOLDER = "data/CheXpert/"


def readCSV(inpFile,delimiter=","):
    with open(inpFile, 'r') as read_obj: return list(csv.reader(read_obj,delimiter=delimiter))

def cropAndScaleImg(img,squareSize):
    width,height = img.size
    cropLen = min(width,height)
    widthMargin = width - cropLen
    heightMargin = height - cropLen
    img = img.crop((widthMargin//2,heightMargin//2,widthMargin//2+cropLen,heightMargin//2+cropLen))
    img = img.resize((squareSize,squareSize),Image.LANCZOS)
    return img

def procImg(tup):
    inpPath,outPath = tup
    if os.path.isfile(outPath):
        return
    os.makedirs(os.path.dirname(outPath),exist_ok=True)
    outImg = cropAndScaleImg(Image.open(inpPath),CROP_SIZE)
    outImg = outImg.convert("L")
    outImg.save(outPath,format='JPEG', subsampling=0, quality=100)

def procImgList(inpLabelFile,outLabelFile,inpSubfolder):
    head,*data = readCSV(CHEXLOCALIZE_FOLDER + inpLabelFile)
    relPaths = [x[0].replace("CheXpert-v1.0/","") for x in data] # paths of the validation set start with "CheXpert-v1.0/valid/"
    imgList = [(CHEXLOCALIZE_FOLDER + inpSubfolder + x.split("/",1)[1], OUT_FOLDER + x) for x in relPaths]
    print("found",len(imgList),"files")
    with mp.Pool() as pool:
        pool.map(procImg,imgList)
    shutil.copy(CHEXLOCALIZE_FOLDER + inpLabelFile, OUT_FOLDER + outLabelFile)
    print("DONE",outLabelFile)


if __name__ == "__main__":
    procImgList("val_labels.csv","valid.csv","val/")
    procImgList("test_labels.csv","test_labels.csv","test/")
