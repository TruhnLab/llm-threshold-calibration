import pickle
import numpy as np
import sklearn
import sklearn.metrics
import matplotlib.pyplot as plt
import os
import math
import json
import multiprocessing as mp

import sklearn.utils
import sklearn.exceptions
from utilFunc import writeCSV
import psutil
import warnings

# precision and F1 are 0 if no sample is predicted positive; sklearn warns about this for every bootstrap sample
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# --- CONFIGURATION & CONSTANTS ---
INP_FOLDER = "output/"
OUT_FOLDER = "evalOutput/"
os.makedirs(OUT_FOLDER, exist_ok=True)

INP_FILE_LIST = [(INP_FOLDER + x, x.replace(".pkl", "")) for x in os.listdir(INP_FOLDER) if "eval_" in x]
INP_FILE_LIST.sort(reverse=False)
print(INP_FILE_LIST)

BOOTSTRAP_N = 10000

PRINT_PRECISSION = 15

USED_CORES = mp.cpu_count()



def logProbsToNormYesProb(inpDict):
    assert len(inpDict.keys()) == 2
    logProbList = [inpDict["no"],inpDict["yes"]]
    if logProbList[0] == logProbList[1]: return 0.5
    probList = [math.exp(x) for x in logProbList]
    probSum = sum(probList)
    normProbList = [x/probSum for x in probList]
    yesProb = normProbList[1]
    return yesProb

def logProbsToUnnormalizedProb(inpDict):
    return math.exp(inpDict["yes"])

def toBinary(inp, threshold=0.5):
    return np.array(inp >= threshold, dtype=np.float32)

def jsonToProb(inpJson):
    prob = float(json.loads(inpJson)["probability"])
    return prob

def applyOneEvalStep(inpTup):
    metric,targetVect,predVect,i = inpTup
    sampleTargetVect, samplePredVect = sklearn.utils.resample(targetVect,predVect,random_state=i)
    out = metric(sampleTargetVect, samplePredVect)
    return out

def evalOneLabel(metric,targetVect,predVect):
    #apply bootstrap
    argList = [(metric,targetVect,predVect,i) for i in range(BOOTSTRAP_N)]
    with mp.Pool(USED_CORES) as pool:
        bootstrapList = pool.map(applyOneEvalStep,argList)
    bootstrapArr = np.array(bootstrapList,dtype=np.float32)

    outVal = metric(targetVect,predVect)
    return outVal,bootstrapArr

def specificity_score(yTarget, yPred):
    out = sklearn.metrics.confusion_matrix(yTarget, yPred).ravel()
    if len(out) != 4: return np.nan
    tn, fp, fn, tp = out
    specificity = tn / (tn+fp)
    return specificity

def sensitivity_plus_specificity(yTarget, yPred):
    sens = sklearn.metrics.recall_score(yTarget, yPred)
    spec = specificity_score(yTarget, yPred)
    return sens + spec

def getMeanAndAvgArr(resultDict,probType,labelList,metricName):
    arrList = [resultDict[(probType,label,metricName)][1] for label in labelList]
    avgArr = np.nanmean(arrList,axis=0)
    totalMean = np.nanmean(avgArr)
    return totalMean,avgArr

def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return sklearn.metrics.roc_auc_score(y_true, y_score)


def calcStats(resultDict,labelNames,propTypeList,fileName):
    print("startStats")
    outCSVList = list()
    
    # Add header
    outCSVList.append(["prob_type", "label", "metric", "value", "lower_95", "upper_95"])
    
    #all Stats
    for key in resultDict.keys():
        outVal, bootstrapArr = resultDict[key]
        lower95,upper95 = np.nan,np.nan
        if not np.isnan(bootstrapArr).all():
            lower95,upper95 = get95CI(bootstrapArr)
        print(*key,round(outVal,PRINT_PRECISSION),"[",round(lower95,PRINT_PRECISSION),",",round(upper95,PRINT_PRECISSION),"]")
        
        # Add to CSV list
        outCSVList.append([key[0], key[1], key[2], round(outVal,PRINT_PRECISSION), round(lower95,PRINT_PRECISSION), round(upper95,PRINT_PRECISSION)])
    
    aurocArrDict = dict()
    for probType in propTypeList:
        meanVal,avgArr = getMeanAndAvgArr(resultDict,probType,labelNames,"testAuroc")
        aurocArrDict[probType] = avgArr
        lower95,upper95 = get95CI(avgArr)
        print(probType,"AVERAGE auroc",round(meanVal,PRINT_PRECISSION),"[",round(lower95,PRINT_PRECISSION),",",round(upper95,PRINT_PRECISSION),"]")
        
        outCSVList.append([probType, "AVERAGE", "auroc", round(meanVal,PRINT_PRECISSION), round(lower95,PRINT_PRECISSION), round(upper95,PRINT_PRECISSION)])

    for probType in propTypeList:
        if probType == "norm": continue
        diffArr = aurocArrDict["norm"] - aurocArrDict[probType]
        diffArr = diffArr[~np.isnan(diffArr)]
        n = diffArr.size
        if n == 0:
            twoSidedP = np.nan
        else:
            p_lo = (np.sum(diffArr <= 0) + 1) / (n + 1)
            p_hi = (np.sum(diffArr >= 0) + 1) / (n + 1)
            twoSidedP = 2 * min(p_lo, p_hi)
        print("norm vs", probType, twoSidedP)
        alpha = 0.05
        diff_lo, diff_hi = np.quantile(diffArr, [alpha/2, 1 - alpha/2])
        print("CI(diff): [", diff_lo, ",", diff_hi, "]")
        
        outCSVList.append(["comparison", f"norm_vs_{probType}", "two_sided_p", round(twoSidedP,PRINT_PRECISSION), "", ""])
        outCSVList.append(["comparison", f"norm_vs_{probType}", "ci_diff", round(np.mean(diffArr),PRINT_PRECISSION), round(diff_lo,PRINT_PRECISSION), round(diff_hi,PRINT_PRECISSION)])
    
    outFile = os.path.join(OUT_FOLDER, fileName, fileName + "_results.csv")
    writeCSV(outCSVList, outFile)
    print(f"Results saved to {outFile}")
    return outCSVList



def get95CI(bootstrapArr, alpha=0.05):
    arr = np.asarray(bootstrapArr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo, hi = np.quantile(arr, [alpha/2, 1 - alpha/2])
    return float(lo), float(hi)




#####################################################################################
#Everything above this line (and the other code in this repro is handcrafted)
#Stuff below this line is partly written by AI (the plotting)
#####################################################################################



COLORS = {
    # pastel palette of the paper figures
    "norm": "#7AAD84",
    "unnorm": "#95C7F2",
    "text": "#FFCDA9",
    "background": "#F5F5F5",
    "grid": "#E0E0E0",
}

BAR_EDGE_COLORS = {  # darker shade of each bar fill
    "norm": "#3A6A45",
    "unnorm": "#5382AA",
    "text": "#BC8865",
}

LINE_COLORS = {  # ROC curve colors (original)
    "norm": "#2E86AB",
    "unnorm": "#A23B72",
    "text": "#F18F01",
}

PLOT_LABELS = {
    "norm": "Normalized",
    "unnorm": "Unnormalized",
    "text": "Text",
}


def build_metrics_lookup(outCSVList):
    """Create a lookup dict for metric values stored in the CSV rows."""
    lookup = {}
    for row in outCSVList[1:]:  # skip header
        try:
            prob_type, label, metric, value, *_ = row
            if value == "" or value is None:
                continue
            value_f = float(value)
            lookup[(str(prob_type), str(label), str(metric))] = value_f
            if metric == "vaildOptThreshold":
                lookup[(str(prob_type), str(label), "optThreshold")] = value_f
        except Exception:
            continue
    return lookup


def _get_metric(lookup, prob_type, label, metric):
    return lookup.get((str(prob_type), str(label), str(metric)), np.nan)


def _sens_spec_text(lookup, prob_type, label, suffix):
    if suffix not in ("_thr_05", "_thr_validOpt"):
        suffix = "_thr_05"

    sens_metric = f"recall_score{suffix}"
    spec_metric = f"specificity_score{suffix}"
    sens = _get_metric(lookup, prob_type, label, sens_metric)
    spec = _get_metric(lookup, prob_type, label, spec_metric)

    sens_str = "N/A" if np.isnan(sens) else f"{sens:.3f}"
    spec_str = "N/A" if np.isnan(spec) else f"{spec:.3f}"

    if suffix == "_thr_05":
        prefix = "Threshold 0.5"
    else:
        prefix = "Optimal Threshold"

    return f"{prefix} \n(Sens.={sens_str}, Spec.={spec_str})"


def bootstrap_roc_curve(args):
    y_target, y_pred, random_state = args
    try:
        sample_target, sample_pred = sklearn.utils.resample(
            y_target, y_pred, random_state=random_state
        )
        if len(np.unique(sample_target)) < 2:
            return None, None
        fpr, tpr, _ = sklearn.metrics.roc_curve(sample_target, sample_pred)
        return fpr, tpr
    except Exception:
        return None, None


def _resolve_opt_threshold(roc_thresholds, valid_threshold):
    if valid_threshold is not None and not np.isnan(valid_threshold):
        return float(valid_threshold)
    best_idx = np.argmax(roc_thresholds[1] - roc_thresholds[0])
    return float(roc_thresholds[2][best_idx])


def generateSingleROC(
    target_vect,
    pred_vect,
    label_name,
    prob_type,
    file_name,
    valid_threshold,
    metrics_lookup=None,
    bootstrap_n=BOOTSTRAP_N,
):
    fig, ax = plt.subplots(figsize=(5, 5))

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(target_vect, pred_vect)
    roc_auc = safe_auc(target_vect, pred_vect)
    opt_threshold_calc = thresholds[np.argmax(tpr - fpr)]
    opt_threshold = (
        float(valid_threshold)
        if valid_threshold is not None and not np.isnan(valid_threshold)
        else float(opt_threshold_calc)
    )

    ax.plot(fpr, tpr, label=f"AUROC = {roc_auc:.3f}", color=LINE_COLORS[prob_type], lw=2)

    if bootstrap_n > 0 and len(target_vect) > 10:
        arg_list = [(target_vect, pred_vect, i) for i in range(bootstrap_n)]
        with mp.Pool(USED_CORES) as pool:
            results = pool.map(bootstrap_roc_curve, arg_list)

        base_fpr = np.linspace(0, 1, 101)
        tpr_list = []
        valid_count = 0
        for b_fpr, b_tpr in results:
            if b_fpr is None or b_tpr is None or len(b_fpr) < 2:
                continue
            try:
                tpr_interp = np.interp(base_fpr, b_fpr, b_tpr)
                tpr_interp[0] = 0.0
                tpr_list.append(tpr_interp)
                valid_count += 1
            except Exception:
                continue

        if valid_count >= max(10, bootstrap_n // 10):
            tpr_array = np.array(tpr_list)
            tpr_lower = np.percentile(tpr_array, 2.5, axis=0)
            tpr_upper = np.percentile(tpr_array, 97.5, axis=0)
            ax.fill_between(base_fpr, tpr_lower, tpr_upper, color="grey", alpha=0.3, label="95% CI")

    binary_pred_05 = toBinary(pred_vect, 0.5)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(target_vect, binary_pred_05).ravel()
    tpr_05, fpr_05 = tp / (tp + fn), fp / (fp + tn)
    thr05_label = (
        _sens_spec_text(metrics_lookup, prob_type, label_name, "_thr_05")
        if metrics_lookup is not None
        else "Threshold 0.5"
    )
    ax.plot(fpr_05, tpr_05, "bo", label=thr05_label, markersize=8)

    binary_pred_opt = toBinary(pred_vect, opt_threshold)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(target_vect, binary_pred_opt).ravel()
    tpr_opt, fpr_opt = tp / (tp + fn), fp / (fp + tn)

    csv_opt_threshold = (
        _get_metric(metrics_lookup, prob_type, label_name, "optThreshold")
        if metrics_lookup is not None
        else np.nan
    )
    threshold_display = (
        csv_opt_threshold
        if csv_opt_threshold is not None and not np.isnan(csv_opt_threshold)
        else opt_threshold
    )

    opt_base_label = (
        _sens_spec_text(metrics_lookup, prob_type, label_name, "_thr_validOpt")
        if metrics_lookup is not None
        else "Optimal Threshold"
    )
    opt_label = f"{opt_base_label}\n (Thr={threshold_display:.3f})"
    ax.plot(fpr_opt, tpr_opt, "o", color="orange", label=opt_label, markersize=8)

    out_path = f"{OUT_FOLDER}{file_name}/singleROC_{file_name}_{prob_type}_{label_name.replace(' ', '_')}.svg"

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True)

    plt.savefig(out_path)
    plt.close()


def generateMultiROC(pred_map, label_name, file_name, metrics_lookup=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = []
    has_ci = False

    for prob_type, (pred_vect, target_vect, valid_threshold) in pred_map.items():
        if len(target_vect) == 0 or len(np.unique(target_vect)) < 2:
            continue

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(target_vect, pred_vect)
        roc_auc = safe_auc(target_vect, pred_vect)
        opt_threshold_calc = thresholds[np.argmax(tpr - fpr)]
        opt_threshold = (
            float(valid_threshold)
            if valid_threshold is not None and not np.isnan(valid_threshold)
            else float(opt_threshold_calc)
        )

        ax.plot(fpr, tpr, color=COLORS[prob_type], lw=2)
        legend_handles.append(
            Patch(
                facecolor=COLORS[prob_type],
                edgecolor=COLORS[prob_type],
                label=f"{PLOT_LABELS[prob_type]} (AUROC = {roc_auc:.3f})",
            )
        )

        if len(target_vect) > 10:
            arg_list = [(target_vect, pred_vect, i) for i in range(BOOTSTRAP_N)]
            with mp.Pool(USED_CORES) as pool:
                results = pool.map(bootstrap_roc_curve, arg_list)

            base_fpr = np.linspace(0, 1, 101)
            tpr_list = []
            valid_count = 0
            for b_fpr, b_tpr in results:
                if b_fpr is None or b_tpr is None or len(b_fpr) < 2:
                    continue
                try:
                    tpr_interp = np.interp(base_fpr, b_fpr, b_tpr)
                    tpr_interp[0] = 0.0
                    tpr_list.append(tpr_interp)
                    valid_count += 1
                except Exception:
                    continue

            if valid_count >= max(10, BOOTSTRAP_N // 10):
                tpr_array = np.array(tpr_list)
                tpr_lower = np.percentile(tpr_array, 2.5, axis=0)
                tpr_upper = np.percentile(tpr_array, 97.5, axis=0)
                ax.fill_between(base_fpr, tpr_lower, tpr_upper, color=COLORS[prob_type], alpha=0.2)
                has_ci = True

        binary_pred_05 = toBinary(pred_vect, 0.5)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(target_vect, binary_pred_05).ravel()
        tpr_05, fpr_05 = tp / (tp + fn), fp / (fp + tn)
        thr05_label = (
            _sens_spec_text(metrics_lookup, prob_type, label_name, "_thr_05")
            if metrics_lookup is not None
            else "Threshold 0.5"
        )
        ax.plot(fpr_05, tpr_05, "s", color=COLORS[prob_type], markersize=6, label=thr05_label)

        binary_pred_opt = toBinary(pred_vect, opt_threshold)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(target_vect, binary_pred_opt).ravel()
        tpr_opt, fpr_opt = tp / (tp + fn), fp / (fp + tn)
        opt_label = (
            _sens_spec_text(metrics_lookup, prob_type, label_name, "_thr_validOpt")
            if metrics_lookup is not None
            else "Optimal Threshold"
        )
        ax.plot(fpr_opt, tpr_opt, "o", color=COLORS[prob_type], markersize=8, label=opt_label)

    if has_ci:
        legend_handles.append(
            Patch(facecolor="grey", alpha=0.2, edgecolor="grey", label="95% CI")
        )
    from matplotlib.lines import Line2D

    legend_handles.append(
        Line2D([0], [0], marker="s", color="grey", linestyle="None", markersize=6, label="Threshold 0.5")
    )
    legend_handles.append(
        Line2D([0], [0], marker="o", color="grey", linestyle="None", markersize=8, label="Optimal Threshold")
    )

    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["grid"])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(handles=legend_handles, loc="lower right")
    ax.grid(True, alpha=0.5)

    out_path = f"{OUT_FOLDER}{file_name}/multiROC_{file_name}_{label_name.replace(' ', '_')}.svg"
    plt.savefig(out_path)
    plt.close()


def generateAUROCBarplot(resultDict, labelNames, probTypes, fileName, outCSVList):
    clean_labels = [name for name in labelNames if name != ""]
    if not clean_labels:
        return

    csv_data = {}
    for row in outCSVList[1:]:
        prob_type, label, metric, value, lower_95, upper_95 = row
        if metric not in ("testAuroc", "auroc"):
            continue
        try:
            value_f = float(value) if value != "" else np.nan
            lower_f = float(lower_95) if lower_95 != "" else np.nan
            upper_f = float(upper_95) if upper_95 != "" else np.nan
        except Exception:
            value_f = lower_f = upper_f = np.nan
        csv_data[(prob_type, label, metric)] = {
            "value": value_f,
            "lower": lower_f,
            "upper": upper_f,
        }

    label_data = []
    for label in clean_labels:
        aurocs = {}
        lower_cis = {}
        upper_cis = {}
        for prob_type in probTypes:
            key = (prob_type, label, "testAuroc")
            if key in csv_data:
                entry = csv_data[key]
                aurocs[prob_type] = entry["value"]
                lower_cis[prob_type] = entry["lower"]
                upper_cis[prob_type] = entry["upper"]
            else:
                aurocs[prob_type] = np.nan
                lower_cis[prob_type] = np.nan
                upper_cis[prob_type] = np.nan
        label_data.append({
            "label": label,
            "aurocs": aurocs,
            "lower_cis": lower_cis,
            "upper_cis": upper_cis,
        })

    label_data.sort(
        key=lambda x: x["aurocs"].get("norm", np.nan)
        if not np.isnan(x["aurocs"].get("norm", np.nan))
        else -1,
        reverse=True,
    )

    avg_entry = {"label": "average", "aurocs": {}, "lower_cis": {}, "upper_cis": {}}
    for prob_type in probTypes:
        key = (prob_type, "AVERAGE", "auroc")
        if key in csv_data:
            entry = csv_data[key]
            avg_entry["aurocs"][prob_type] = entry["value"]
            avg_entry["lower_cis"][prob_type] = entry["lower"]
            avg_entry["upper_cis"][prob_type] = entry["upper"]
        else:
            avg_entry["aurocs"][prob_type] = np.nan
            avg_entry["lower_cis"][prob_type] = np.nan
            avg_entry["upper_cis"][prob_type] = np.nan
    label_data.insert(0, avg_entry)

    fig, ax = plt.subplots(figsize=(max(8, len(label_data) * 0.62), 5.5))  # same text size as the threshold plot after scaling
    x = np.arange(len(label_data))
    width = 0.25

    for i, prob_type in enumerate(probTypes):
        aurocs = [entry["aurocs"].get(prob_type, np.nan) for entry in label_data]
        lower_cis = [entry["lower_cis"].get(prob_type, np.nan) for entry in label_data]
        upper_cis = [entry["upper_cis"].get(prob_type, np.nan) for entry in label_data]

        yerr_lower = [
            (aurocs[j] - lower_cis[j]) if not np.isnan(aurocs[j]) and not np.isnan(lower_cis[j]) else 0
            for j in range(len(aurocs))
        ]
        yerr_upper = [
            (upper_cis[j] - aurocs[j]) if not np.isnan(aurocs[j]) and not np.isnan(upper_cis[j]) else 0
            for j in range(len(aurocs))
        ]

        yerr = [yerr_lower, yerr_upper]

        bars = ax.bar(
            x + i * width - width,
            aurocs,
            width,
            label=PLOT_LABELS[prob_type],
            color=COLORS[prob_type],
            edgecolor=BAR_EDGE_COLORS[prob_type],
            linewidth=0.8,
            yerr=yerr,
            capsize=3,
            error_kw={"alpha": 0.7},
        )

        for j, bar in enumerate(bars):
            if not np.isnan(aurocs[j]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.05,
                    f"{aurocs[j]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    fontweight="bold",
                )

    ax.set_xlabel("Medical Findings")
    ax.set_ylabel("AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels([entry["label"] for entry in label_data], rotation=45, ha="right")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(probTypes), frameon=False)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.axhline(y=0.5, color="gray", linewidth=1.5, linestyle="--")  # chance level

    out_path = f"{OUT_FOLDER}{fileName}/aurocBarplot_{fileName}.svg"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def generateCombinedMultiROC(propTypeDict, allTestTargets, labelNames, fileName, metrics_lookup=None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    valid_items = []
    for i, label_name in enumerate(labelNames):
        if label_name == "":
            continue
        pred_map = {}
        for prob_type, (_, test_arr) in propTypeDict.items():
            mask = ~np.isnan(test_arr[:, i])
            target_vect = allTestTargets[:, i][mask]
            pred_vect = test_arr[:, i][mask]
            if len(target_vect) == 0 or len(np.unique(target_vect)) < 2:
                continue
            valid_threshold = (
                _get_metric(metrics_lookup, prob_type, label_name, "vaildOptThreshold")
                if metrics_lookup is not None
                else np.nan
            )
            pred_map[prob_type] = (pred_vect, target_vect, valid_threshold)
        if pred_map:
            valid_items.append((i, label_name, pred_map))

    if not valid_items:
        return

    n_plots = len(valid_items)
    cols = min(3, n_plots)
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for plot_idx, (_, label_name, pred_map) in enumerate(valid_items):
        ax = axes[plot_idx]

        legend_handles = []
        has_ci = False

        for prob_type, (pred_vect, target_vect, valid_threshold) in pred_map.items():
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(target_vect, pred_vect)
            roc_auc = safe_auc(target_vect, pred_vect)
            opt_threshold_calc = thresholds[np.argmax(tpr - fpr)]
            opt_threshold = (
                float(valid_threshold)
                if valid_threshold is not None and not np.isnan(valid_threshold)
                else float(opt_threshold_calc)
            )

            ax.plot(fpr, tpr, color=COLORS[prob_type], lw=2)
            legend_handles.append(
                Patch(
                    facecolor=COLORS[prob_type],
                    edgecolor=COLORS[prob_type],
                    label=f"{PLOT_LABELS[prob_type]} (AUROC = {roc_auc:.3f})",
                )
            )

            valid_count = 0
            if len(target_vect) > 10:
                arg_list = [(target_vect, pred_vect, rs) for rs in range(BOOTSTRAP_N)]
                with mp.Pool(USED_CORES) as pool:
                    results = pool.map(bootstrap_roc_curve, arg_list)

                base_fpr = np.linspace(0, 1, 101)
                tpr_list = []
                for b_fpr, b_tpr in results:
                    if b_fpr is None or b_tpr is None or len(b_fpr) < 2:
                        continue
                    try:
                        tpr_interp = np.interp(base_fpr, b_fpr, b_tpr)
                        tpr_interp[0] = 0.0
                        tpr_list.append(tpr_interp)
                        valid_count += 1
                    except Exception:
                        continue

                if valid_count >= max(10, BOOTSTRAP_N // 10):
                    tpr_array = np.array(tpr_list)
                    tpr_lower = np.percentile(tpr_array, 2.5, axis=0)
                    tpr_upper = np.percentile(tpr_array, 97.5, axis=0)
                    ax.fill_between(base_fpr, tpr_lower, tpr_upper, color=COLORS[prob_type], alpha=0.2)
                    has_ci = True

            binary_pred_05 = toBinary(pred_vect, 0.5)
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(target_vect, binary_pred_05).ravel()
            tpr_05, fpr_05 = tp / (tp + fn), fp / (fp + tn)
            thr05_label = (
                _sens_spec_text(metrics_lookup, prob_type, label_name, "_thr_05")
                if metrics_lookup is not None
                else "Threshold 0.5"
            )
            ax.plot(fpr_05, tpr_05, "s", color=COLORS[prob_type], markersize=6, label=thr05_label)

            binary_pred_opt = toBinary(pred_vect, opt_threshold)
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(target_vect, binary_pred_opt).ravel()
            tpr_opt, fpr_opt = tp / (tp + fn), fp / (fp + tn)
            opt_label = (
                _sens_spec_text(metrics_lookup, prob_type, label_name, "_thr_validOpt")
                if metrics_lookup is not None
                else "Optimal Threshold"
            )
            ax.plot(fpr_opt, tpr_opt, "o", color=COLORS[prob_type], markersize=8, label=opt_label)

        if has_ci:
            legend_handles.append(
                Patch(facecolor="grey", alpha=0.2, edgecolor="grey", label="95% CI")
            )
        legend_handles.append(
            Line2D([0], [0], marker="s", color="grey", linestyle="None", markersize=6, label="Threshold 0.5")
        )
        legend_handles.append(
            Line2D([0], [0], marker="o", color="grey", linestyle="None", markersize=8, label="Optimal Threshold")
        )

        ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["grid"])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(label_name)
        ax.legend(handles=legend_handles, loc="lower right")
        ax.grid(True, alpha=0.5)

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    out_path = f"{OUT_FOLDER}{fileName}/combinedMultiROC_{fileName}.svg"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def generateCombinedSingleROC(propTypeDict, allTestTargets, labelNames, probType, fileName, metrics_lookup=None):
    clean_labels = [name for name in labelNames if name != ""]
    if not clean_labels:
        return

    n_plots = len(clean_labels)
    cols = min(3, n_plots)
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    valid_arr, test_arr = propTypeDict[probType]

    for plot_idx, (i, label_name) in enumerate([(i, name) for i, name in enumerate(labelNames) if name != ""]):
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]

        mask = ~np.isnan(test_arr[:, i])
        target_vect = allTestTargets[:, i][mask]
        pred_vect = test_arr[:, i][mask]

        if len(target_vect) == 0 or len(np.unique(target_vect)) <= 1:
            ax.set_visible(False)
            continue

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(target_vect, pred_vect)
        roc_auc = safe_auc(target_vect, pred_vect)
        opt_threshold_calc = thresholds[np.argmax(tpr - fpr)]
        valid_threshold = (
            _get_metric(metrics_lookup, probType, label_name, "vaildOptThreshold")
            if metrics_lookup is not None
            else np.nan
        )
        opt_threshold = (
            float(valid_threshold)
            if valid_threshold is not None and not np.isnan(valid_threshold)
            else float(opt_threshold_calc)
        )

        ax.plot(fpr, tpr, label=f"AUROC = {roc_auc:.3f}", color=COLORS[probType], lw=2)

        has_ci = False
        if len(target_vect) > 10:
            arg_list = [(target_vect, pred_vect, j) for j in range(100)]
            with mp.Pool(USED_CORES) as pool:
                results = pool.map(bootstrap_roc_curve, arg_list)

            base_fpr = np.linspace(0, 1, 101)
            tpr_list = []
            valid_count = 0
            for b_fpr, b_tpr in results:
                if b_fpr is None or b_tpr is None or len(b_fpr) < 2:
                    continue
                try:
                    tpr_interp = np.interp(base_fpr, b_fpr, b_tpr)
                    tpr_interp[0] = 0.0
                    tpr_list.append(tpr_interp)
                    valid_count += 1
                except Exception:
                    continue

            if valid_count >= 10:
                tpr_array = np.array(tpr_list)
                tpr_lower = np.percentile(tpr_array, 2.5, axis=0)
                tpr_upper = np.percentile(tpr_array, 97.5, axis=0)
                ax.fill_between(base_fpr, tpr_lower, tpr_upper, color="grey", alpha=0.3, label="95% CI")
                has_ci = True

        binary_pred_05 = toBinary(pred_vect, 0.5)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(target_vect, binary_pred_05).ravel()
        tpr_05, fpr_05 = tp / (tp + fn), fp / (fp + tn)
        thr05_label = (
            _sens_spec_text(metrics_lookup, probType, label_name, "_thr_05")
            if metrics_lookup is not None
            else "Threshold 0.5"
        )
        ax.plot(fpr_05, tpr_05, "bo", label=thr05_label, markersize=6)

        binary_pred_opt = toBinary(pred_vect, opt_threshold)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(target_vect, binary_pred_opt).ravel()
        tpr_opt, fpr_opt = tp / (tp + fn), fp / (fp + tn)
        csv_opt_threshold = (
            _get_metric(metrics_lookup, probType, label_name, "optThreshold")
            if metrics_lookup is not None
            else np.nan
        )
        threshold_display = (
            csv_opt_threshold
            if csv_opt_threshold is not None and not np.isnan(csv_opt_threshold)
            else opt_threshold
        )
        opt_base_label = (
            _sens_spec_text(metrics_lookup, probType, label_name, "_thr_validOpt")
            if metrics_lookup is not None
            else "Optimal Threshold"
        )
        opt_label = f"{opt_base_label}\n (Thr={threshold_display:.3f})"
        ax.plot(fpr_opt, tpr_opt, "o", color="orange", label=opt_label, markersize=6)

        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(label_name)
        ax.legend(loc="lower right")
        ax.grid(True)

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    out_path = f"{OUT_FOLDER}{fileName}/combinedSingleROC_{fileName}_{probType}.svg"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def generateAUROCPlot(propTypeDict, allTestTargets, labelNames, resultDict, fileName, outCSVList):
    print("Generating AUROC plots...")
    probTypes = list(propTypeDict.keys())

    metrics_lookup = build_metrics_lookup(outCSVList)

    for i, label_name in enumerate(labelNames):
        if label_name == "":
            continue
        for prob_type, (_, test_arr) in propTypeDict.items():
            mask = ~np.isnan(test_arr[:, i])
            target_vect = allTestTargets[:, i][mask]
            pred_vect = test_arr[:, i][mask]
            if len(target_vect) == 0 or len(np.unique(target_vect)) < 2:
                continue
            valid_threshold = _get_metric(metrics_lookup, prob_type, label_name, "vaildOptThreshold")
            generateSingleROC(
                target_vect,
                pred_vect,
                label_name,
                prob_type,
                fileName,
                valid_threshold,
                metrics_lookup=metrics_lookup,
            )

    for i, label_name in enumerate(labelNames):
        if label_name == "":
            continue
        pred_map = {}
        for prob_type, (_, test_arr) in propTypeDict.items():
            mask = ~np.isnan(test_arr[:, i])
            target_vect = allTestTargets[:, i][mask]
            pred_vect = test_arr[:, i][mask]
            if len(target_vect) == 0 or len(np.unique(target_vect)) < 2:
                continue
            valid_threshold = _get_metric(metrics_lookup, prob_type, label_name, "vaildOptThreshold")
            pred_map[prob_type] = (pred_vect, target_vect, valid_threshold)
        if pred_map:
            generateMultiROC(pred_map, label_name, fileName, metrics_lookup=metrics_lookup)

    generateAUROCBarplot(resultDict, labelNames, probTypes, fileName, outCSVList)

    generateCombinedMultiROC(propTypeDict, allTestTargets, labelNames, fileName, metrics_lookup=metrics_lookup)
    for prob_type in probTypes:
        generateCombinedSingleROC(
            propTypeDict,
            allTestTargets,
            labelNames,
            prob_type,
            fileName,
            metrics_lookup=metrics_lookup,
        )

    print("AUROC plots generated successfully!")

def evalFile(inpFilePath,thresholdMetricList,fileName):
    file_out_folder = os.path.join(OUT_FOLDER, fileName)
    if os.path.exists(file_out_folder):
        print("SKIP",file_out_folder)
        return
    os.makedirs(file_out_folder, exist_ok=False)

    with open(inpFilePath, "rb") as f:
        dataDict = pickle.load(f)
    
    labelNames = dataDict["labelNames"]
    print(labelNames)

    def extractProb(func,key):
        return np.array([[func(y) for y in x] for x in dataDict[key]], dtype=np.float32)

    #label for test

    allTestTargets = np.array(dataDict["testTargets"], dtype=np.int32)
    
    allTestNormProbs = extractProb(logProbsToNormYesProb, "testLogProbs")
    allTestUnnormProbs = extractProb(logProbsToUnnormalizedProb, "testLogProbs")
    allTestTextProbs = extractProb(jsonToProb, "testJsonProbList")

    allVaildTargets = np.array(dataDict["validTargets"], dtype=np.int32)

    allValidNormProbs = extractProb(logProbsToNormYesProb, "validLogProbs")
    allValidUnnormProbs = extractProb(logProbsToUnnormalizedProb, "validLogProbs")
    allValidTextProbs = extractProb(jsonToProb, "validJsonProbList")

    propTypeDict = {
        "norm":(allValidNormProbs,allTestNormProbs),
        "unnorm":(allValidUnnormProbs,allTestUnnormProbs),
        "text":(allValidTextProbs,allTestTextProbs),
    }

    resultDict = dict()

    for probType,(vaildProbArr,testProbArr) in propTypeDict.items():
        for i, name in enumerate(labelNames):

            validTargetVect = allVaildTargets[:,i][~np.isnan(vaildProbArr[:,i])]
            validPredVect = vaildProbArr[:,i][~np.isnan(vaildProbArr[:,i])]
            resultDict[(probType,name,"vaildOldLen")] = (len(vaildProbArr[:,i]),np.nan)
            resultDict[(probType,name,"vaildNewLen")] = (len(validPredVect),np.nan)

            testTargetVect = allTestTargets[:,i][~np.isnan(testProbArr[:,i])]
            testPredVect = testProbArr[:,i][~np.isnan(testProbArr[:,i])]
            resultDict[(probType,name,"testOldLen")] = (len(testProbArr[:,i]),np.nan)
            resultDict[(probType,name,"testNewLen")] = (len(testPredVect),np.nan)

            # vaildAuroc, vaildBootstrapAUROC = evalOneLabel(safe_auc,validTargetVect,validPredVect)
            # resultDict[(probType,name,"vaildAuroc")] = (vaildAuroc, vaildBootstrapAUROC)

            testAuroc, testBootstrapAUROC = evalOneLabel(safe_auc,testTargetVect,testPredVect)
            resultDict[(probType,name,"testAuroc")] = (testAuroc, testBootstrapAUROC)

            vaildFpr, vaildTpr, vaildThresholds = sklearn.metrics.roc_curve(validTargetVect,validPredVect, drop_intermediate=False)
            vaildOptThreshold = vaildThresholds[np.argmax(vaildTpr - vaildFpr)]
            resultDict[(probType,name,"vaildOptThreshold")] = (vaildOptThreshold, np.nan)

            def applyTestMetricsList(threshold,nameAdd):
                for metric in thresholdMetricList:
                    outVal, bootstrapArr = evalOneLabel(metric,testTargetVect,toBinary(testPredVect,threshold=threshold))
                    metricName = metric.__name__ + nameAdd
                    resultDict[(probType,name,metricName)] = (outVal, bootstrapArr)
                    print(*(probType,name,metricName),outVal)
            
            applyTestMetricsList(threshold=0.5,nameAdd="_thr_05")
            applyTestMetricsList(threshold=vaildOptThreshold,nameAdd="_thr_validOpt")
    
    outCSVList = calcStats(resultDict,labelNames,propTypeDict.keys(),fileName)

    generateAUROCPlot(propTypeDict, allTestTargets, labelNames, resultDict, fileName, outCSVList) # TODO



if __name__ == "__main__":
    thresholdMetricList = [
        sklearn.metrics.accuracy_score,
        sklearn.metrics.f1_score, 
        sklearn.metrics.precision_score,
        sklearn.metrics.recall_score,
        specificity_score,
        sensitivity_plus_specificity,
    ]
    parent = psutil.Process()
    parent.nice(19) # works on linux, might throw an error on non Linux system => giv processes a low priority
    print(INP_FILE_LIST)
    for inpFile in INP_FILE_LIST:
        filePath,name = inpFile
        evalFile(filePath,thresholdMetricList,name)