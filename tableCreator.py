import os
from utilFunc import readCSV
from functools import cache
INP_FOLDER = "evalOutput/"

MODEL = "_MedGemma27B_"
TAB3_MODEL_LIST = ["_MedGemma27B_","_MedGemma4B_","_Gemma27B_","_Gemma4B_"]

ROUNDING_NUM = 3

DATASET_LIST = ["CheXpert","DERM12345","G1020","HAM10000","MIMIC","NCT_CRC_HE"]

Table1Template = """"""

def getEvalCSVList(model):
    evalFolderList = [INP_FOLDER + x + "/" for x in os.listdir(INP_FOLDER)]
    evalFolderList = [x for x in evalFolderList if model in x]
    return [x + [y for y in os.listdir(x) if ".csv" in y][0] for x in evalFolderList]

evalCSVList = getEvalCSVList(MODEL)

def getLine(data,*features):
    for oneFeature in features:
        data = [x for x in data if oneFeature in x]
    assert len(data) == 1
    return data[0]

@cache
def createTabLine(inpCSV):
    data = readCSV(inpCSV)

    rawRetDict = {
        "norm":getLine(data,"norm","AVERAGE","auroc")[3:],
        "unnorm":getLine(data,"unnorm","AVERAGE","auroc")[3:],
        "text":getLine(data,"text","AVERAGE","auroc")[3:],
        ("compP","unnorm"):getLine(data,"comparison","norm_vs_unnorm","two_sided_p")[3:4],
        ("compP","text"):getLine(data,"comparison","norm_vs_text","two_sided_p")[3:4],
        ("compCI","unnorm"):getLine(data,"comparison","norm_vs_unnorm","ci_diff")[3:],
        ("compCI","text"):getLine(data,"comparison","norm_vs_text","ci_diff")[3:],
            }
    rawRetDict = {k:[float(x) for x in v] for k,v in rawRetDict.items()}
    # CSV stores norm - comparator; the table reports comparator - norm, so negate and swap the CI bounds
    for key in [("compCI","unnorm"),("compCI","text")]:
        mean, low, high = rawRetDict[key]
        rawRetDict[key] = [-mean, -high, -low]
    retDict = {k:[(str(round(x,ROUNDING_NUM))+"0"*5)[:(6 if "-" in str(x) else 5)] for x in v] for k,v in rawRetDict.items()}
    return retDict,rawRetDict

def createTab1(evalCSVList):
    out = ""
    for dataset in DATASET_LIST:
        evalFile = [x for x in evalCSVList if dataset in x]
        assert len(evalFile) == 1
        evalFile = evalFile[0]
        tabDict,_ = createTabLine(evalFile)
        out += dataset.replace("_","-")
        out += " & "
        val = lambda key: tabDict[key][0] + " [" + tabDict[key][1] + ", " + tabDict[key][2] + "]"
        out += val("norm")
        out += " & "
        out += val("unnorm")
        out += " & "
        out += val("text")
        out += " \\\\\n"
    print("++++++++ TABLE 1 +++++++++" + MODEL,out,"++++++++++",sep="\n")
    return out

def multirow(cell):
    """Center a single-line cell vertically across the value line and the CI line below it."""
    return "\\multirow{2}{*}{" + cell + "}"

def createTab2(evalCSVList):
    """Two lines per dataset: difference and p-value, then the CI of the difference below."""
    out = ""
    for dataset in DATASET_LIST:
        evalFile = [x for x in evalCSVList if dataset in x]
        assert len(evalFile) == 1
        evalFile = evalFile[0]
        tabDict,rawTabDict = createTabLine(evalFile)

        def getDiffCells(key):
            pVal = float(rawTabDict[("compP",key)][0])
            bold = (lambda x: "\\textbf{" + x + "}") if pVal < 0.0042 else (lambda x: x)
            pText = "< .001" if pVal < 0.001 else tabDict[("compP",key)][0][1:]
            pCell = "$\\mathbf{" + pText + "}$" if pVal < 0.0042 else "$" + pText + "$"
            ciCell = bold("[" + tabDict[("compCI",key)][1] + ", " + tabDict[("compCI",key)][2] + "]")
            return bold(tabDict[("compCI",key)][0]), pCell, ciCell

        rawDelta, rawP, rawCI = getDiffCells("unnorm")
        textDelta, textP, textCI = getDiffCells("text")
        out += " & ".join([multirow(dataset.replace("_","-")), multirow(tabDict["norm"][0]), rawDelta, multirow(rawP), textDelta, multirow(textP)]) + " \\\\\n"
        out += " & ".join(["", "", rawCI, "", textCI, ""]) + " \\\\\n"
        out += "\\addlinespace\n" if dataset != DATASET_LIST[-1] else ""

    print("++++++++ TABLE 2 +++++++++" + MODEL,out,"++++++++++",sep="\n")
    return out

def createTab3():
    csvListPerModel = [getEvalCSVList(model) for model in TAB3_MODEL_LIST]
    out = ""
    for dataset in DATASET_LIST:
        values, cis = [], []
        for modelCSVList in csvListPerModel:
            evalFile = [x for x in modelCSVList if dataset in x]
            assert len(evalFile) == 1
            tabDict,_ = createTabLine(evalFile[0])
            values.append(tabDict["norm"][0])
            cis.append("[" + tabDict["norm"][1] + ", " + tabDict["norm"][2] + "]")
        out += " & ".join([multirow(dataset.replace("_","-"))] + values) + " \\\\\n"
        out += " & ".join([""] + cis) + " \\\\\n"
        out += "\\addlinespace\n" if dataset != DATASET_LIST[-1] else ""
    print("++++++++ TABLE 3 +++++++++",out,"++++++++++",sep="\n")
    return out









print(evalCSVList)

createTab1(evalCSVList)
createTab2(evalCSVList)
createTab3()


