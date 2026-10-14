"""Generate the prompt, label and per-class AUROC sections of the supplement from the inference pickles
and evaluation CSVs, written as LaTeX to OUT_FILE."""
import pickle
import re
from utilFunc import readCSV
import HAM10000
import DERM12345
import NCT_CRC_HE_100K
import G1020

PICKLE_FOLDER = "output/"
EVAL_FOLDER = "evalOutput/"
OUT_FILE = "evalOutput/supplementTables.tex"
MODEL = "MedGemma27B"
PLACEHOLDER = "{finding}"
ROUNDING_NUM = 3

# (pickle name, name in the paper)
DATASET_LIST = [("CheXpert", "CheXpert"), ("MIMIC", "MIMIC-CXR"), ("HAM10000", "HAM10000"),
                ("DERM12345", "DERM12345"), ("NCT_CRC_HE_100K_1000", "NCT-CRC-HE-100K"), ("G1020", "G1020")]
CHEXPERT_LABELS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
                   "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
                   "Fracture", "Support Devices"]


def tex(text):
    return re.sub(r"([&%_#$])", r"\\\1", text)


def loadPickle(dataset):
    with open(PICKLE_FOLDER + f"eval_{MODEL}_{dataset}.pkl", "rb") as f:
        return pickle.load(f)


def toTemplate(prompts, phrases):
    """Replace the label phrase in every prompt by the placeholder and check that one structure remains."""
    templates = set()
    for prompt, phrase in zip(prompts, phrases):
        assert prompt.count(phrase) == 1, (phrase, prompt)
        templates.add(prompt.replace(phrase, PLACEHOLDER))
    assert len(templates) == 1, templates
    return templates.pop()


def originalLabels(dataset, phrases):
    """Return (level, original label) for every prompt phrase."""
    if dataset in ["CheXpert", "MIMIC"]:
        lookup = {x.lower(): ("", x) for x in CHEXPERT_LABELS}
    elif dataset in ["HAM10000", "DERM12345"]:
        module = HAM10000 if dataset == "HAM10000" else DERM12345
        levels = [module.DIAG_1_LABEL_LIST, module.DIAG_2_LABEL_LIST, module.DIAG_3_LABEL_LIST]
        lookup = {module.LABEL_DESCRIPTIONS[x]: (str(i + 1), x) for i, level in enumerate(levels) for x in level}
    else:
        labelDict = NCT_CRC_HE_100K.labelDict if dataset.startswith("NCT") else G1020.labelDict
        lookup = {v.lower(): ("", k) for k, v in labelDict.items()}
    return [lookup[x] for x in phrases]


def listing(text):
    return "\\begin{lstlisting}\n" + text.rstrip("\n") + "\n\\end{lstlisting}\n"


def promptSection(dataDicts):
    out = "\\section{Prompts}\\label{sec:prompts}\n"
    out += (f"Each label was queried in a separate context. The placeholder \\texttt{{\\{{finding\\}}}} was replaced by the "
            "label phrase listed in Section~\\ref{sec:labels}. The image was passed together with the prompt.\n\n")
    for kind, title in [("yesNo", "Yes/no prompts (normalized and raw token probabilities)"),
                        ("json", "Prompts for text-elicited probabilities")]:
        out += f"\\subsection{{{title}}}\n"
        groups = {}
        for dataset, name in DATASET_LIST:
            dataDict = dataDicts[dataset]
            prompts = dataDict["promptList"] if kind == "yesNo" else [x[0] for x in dataDict["jsonPrompts"]]
            groups.setdefault(toTemplate(prompts, dataDict["labelNames"]), []).append(name)
        for template, names in groups.items():
            out += f"\\noindent\\begin{{minipage}}{{\\linewidth}}\n\\textbf{{{', '.join(names)}}}\n" + listing(template) + "\\end{minipage}\n\\medskip\n\n"
    schemas = {str(x["jsonPrompts"][0][1]) for x in dataDicts.values()}
    assert len(schemas) == 1
    out += ("\\subsection{JSON schema for guided decoding}\n"
            "The text-elicited probability was generated with guided decoding (xgrammar) constrained to the following "
            "JSON schema, which only admits numbers with three decimal places (all observed values lay between 0 and 1):\n")
    out += listing('{"type": "object",\n "properties": {"probability": {"type": "string", "pattern": "[01]\\\\.\\\\d{3}"}},\n "required": ["probability"]}')
    return out


def labelSection(dataDicts):
    out = "\\section{Labels}\\label{sec:labels}\n"
    out += ("For each dataset, the original label name and the phrase inserted into the prompts are listed. "
            "For HAM10000 and DERM12345, the level refers to the hierarchical diagnosis label (1: benign, indeterminate or "
            "malignant; 2 and 3: increasingly specific diagnoses).\n\n")
    for dataset, name in DATASET_LIST:
        phrases = dataDicts[dataset]["labelNames"]
        rows = originalLabels(dataset, phrases)
        withLevel = dataset in ["HAM10000", "DERM12345"]
        out += "\\begin{table}[htbp]\n\\centering\n\\small\n"
        out += f"\\caption{{Labels of {name} ({len(phrases)} labels).}}\\label{{tab:labels_{dataset}}}\n"
        out += "\\begin{tabular}{" + ("cp{0.38\\linewidth}p{0.44\\linewidth}" if withLevel else "p{0.4\\linewidth}p{0.45\\linewidth}") + "}\n\\toprule\n"
        out += ("Level & " if withLevel else "") + "Original label & Phrase in prompt \\\\\n\\midrule\n"
        for (level, original), phrase in zip(rows, phrases):
            out += (f"{level} & " if withLevel else "") + f"{tex(original)} & {tex(phrase)} \\\\\n"
        out += "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
    return out


def aurocSection(dataDicts):
    out = "\\section{Per-class AUROC values}\\label{sec:perclass}\n"
    out += (f"AUROC values with 95\\% confidence intervals on the test set for MedGemma-27B, for the normalized token "
            "probability, the raw token probability and the text-elicited probability.\n\n")
    fmt = lambda x: f"{float(x):.{ROUNDING_NUM}f}"
    for dataset, name in DATASET_LIST:
        csvPath = EVAL_FOLDER + f"eval_{MODEL}_{dataset}/eval_{MODEL}_{dataset}_results.csv"
        values = {(x[0], x[1]): x[3:6] for x in readCSV(csvPath) if len(x) > 2 and x[2] == "testAuroc"}
        phrases = dataDicts[dataset]["labelNames"]
        rows = originalLabels(dataset, phrases)
        out += "\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        out += f"\\caption{{Per-class AUROC values for {name} (MedGemma-27B, test set).}}\\label{{tab:auroc_{dataset}}}\n"
        out += "\\begin{tabular}{p{0.3\\linewidth}ccc}\n\\toprule\n"
        out += "Label & Normalized (95\\% CI) & Raw (95\\% CI) & Text-elicited (95\\% CI) \\\\\n\\midrule\n"
        for (_, original), phrase in zip(rows, phrases):
            cells = []
            for probType in ["norm", "unnorm", "text"]:
                value, low, high = values[(probType, phrase)]
                cells.append("n/a" if value == "nan" else f"{fmt(value)} [{fmt(low)}, {fmt(high)}]")
            out += f"{tex(original)} & " + " & ".join(cells) + " \\\\\n"
        out += "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
    return out


if __name__ == "__main__":
    dataDicts = {dataset: loadPickle(dataset) for dataset, _ in DATASET_LIST}
    generated = promptSection(dataDicts) + "\n" + labelSection(dataDicts) + "\\clearpage\n" + aurocSection(dataDicts)
    with open(OUT_FILE, "w") as f:
        f.write(generated)
    print("written", OUT_FILE)
