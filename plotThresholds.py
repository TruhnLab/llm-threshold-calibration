import os
import utilFunc
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["svg.hashsalt"] = "0" # reproducible SVG files
EVAL_PATH = "evalOutput/"


folderList = [EVAL_PATH + x + "/" for x in os.listdir(EVAL_PATH) if "MedGemma27B" in x]
print(folderList)


def extractThresholds(path):
    csvList = [x for x in os.listdir(path) if x.endswith(".csv")]
    assert len(csvList) == 1
    csvPath = path + csvList[0]
    head,*data = utilFunc.readCSV(csvPath)
    optThresholdList = [(x[1],float(x[3])) for x in data if x[0] == "norm" and x[2] == "vaildOptThreshold" and x[1] != "" and x[3] != "inf"]
    optThresholdList.sort(key=lambda x: abs(0.5-x[1]))
    return optThresholdList

def plotThresholds(path):
    optThresholdList = extractThresholds(path)
    
    if not optThresholdList:
        print(f"No optimal thresholds found for {path}")
        return

    labels, thresholds_str = zip(*optThresholdList)
    thresholds = np.array([float(t) for t in thresholds_str])
    
    # Calculate average
    avg_threshold = np.mean(thresholds)
    
    # Add average to the data
    labels = list(labels) + ['average']
    thresholds = np.append(thresholds, avg_threshold)
    
    # Calculate bar heights (difference from 0.5)
    bar_heights = thresholds - 0.5
    
    fig, ax = plt.subplots(figsize=(max(len(labels) * 0.6, 6), 6))
    
    # Create color array: regular bars in default color, average in orange
    colors = ['#F0D286'] * (len(labels) - 1) + ['#E69138']  # optimal threshold (yellow), average (orange)
    edgecolors = ['#E5B02D'] * (len(labels) - 1) + ['#B45F06']
    
    # Bars start at 0.5 and extend to threshold value
    bars = ax.bar(np.arange(len(labels)), bar_heights, bottom=0.5, color=colors, edgecolor=edgecolors, linewidth=0.8)
    
    ax.set_xlabel('Medical Findings')
    ax.set_ylabel('Optimal Threshold')
    #ax.set_title(f'Optimal Thresholds for norm probability')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.axhline(y=0.5, color='#4789CB', linewidth=1.5, linestyle='--')  # default threshold
    
    for i, (bar, threshold) in enumerate(zip(bars, thresholds)):
        bar_length = abs(bar_heights[i])
        
        # Text inside bar if length > 0.25, otherwise outside
        if bar_length > 0.25:
            # Inside the bar, centered
            y_pos = 0.5 + bar_heights[i] / 2.0
            va = 'center'
            rotation = 90
        else:
            # Outside the bar
            if bar_heights[i] >= 0:
                # Bar goes up, text above
                y_pos = threshold + 0.01
                va = 'bottom'
            else:
                # Bar goes down, text below
                y_pos = threshold - 0.01
                va = 'top'
            rotation = 90
        
        ax.text(bar.get_x() + bar.get_width()/2.0, y_pos, f'{threshold:.3f}', 
                ha='center', va=va, rotation=rotation, fontsize=9, fontweight='bold')

    fileName = os.path.basename(os.path.normpath(path))
    outPath = os.path.join(path, f"optimal_thresholds_{fileName}.svg")
    plt.tight_layout()
    plt.savefig(outPath, metadata={"Date": None})
    #plt.show()
    plt.close(fig)
    print(f"Plot saved to {outPath}")

if __name__ == "__main__":
    for folder in folderList:
        plotThresholds(folder)


