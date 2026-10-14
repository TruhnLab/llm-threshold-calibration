import csv
import gzip
import json

def readCSV(inpFile,delimiter=","):
    with open(inpFile, 'r') as read_obj: return list(csv.reader(read_obj,delimiter=delimiter))

def readCompressedJson(inputFile):
    with gzip.open(inputFile, "rt", encoding="utf-8") as fIn:
        return json.load(fIn)

def writeCSV(data, outFile, delimiter=","):
    with open(outFile, 'w', newline='') as write_obj:
        writer = csv.writer(write_obj, delimiter=delimiter)
        writer.writerows(data)

def readZipIterating(filePath, delimiter=","):
    #input: path to zip file
    #output: iterator for lines in the csv
    with gzip.open(filePath, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            yield row

def readZip(filePath, delimiter=","):
    #input: path to zip file
    #output: iterator for lines in the csv
    with gzip.open(filePath, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        return list(reader)