import numpy as np
import json
import argparse
import matplotlib.pyplot as plt

def parser():
    parser = argparse.ArgumentParser(description="Plots given particle features.")
    parser.add_argument(
        "-d", "--datafile",
        type=str,
        default="featured_data.json",
        help="Path to the input data file."
    )
    parser.add_argument(
        "-t", "--numhitsthreshold",
        type=int,
        default=4,
        help="The minimum number of hits a cluster must have to be considered."
    )
    parser.add_argument(
        "-p", "--densityplot",
        action="store_true",
        default=False,
        help="Whether to plot event densities rather than absolute quantities."
    )
    return parser.parse_args()

def dataUnpack(dataFile):
    with open(f"Data/{dataFile}", "r") as f:
        data = json.load(f)
    
    clusters = np.array([c for c in data])
    print(f"There are {len(clusters)} clusters.")

    return clusters

# Plots histograms of each feature.
def featurePlot(clusters, features, numHitsThreshold, densityPlot):
    numFeatures, plotCols = len(features), 2
    plotRows = (numFeatures+plotCols-1)//plotCols
    fig, axes = plt.subplots(nrows=plotRows, ncols=plotCols, figsize=(10, 5 * plotRows), layout="constrained")
    axes = axes.flatten()

    numHits = np.array([cluster["numPfoHits"] for cluster in clusters])
    PDGCodes = np.array([cluster["PDGCode"] for cluster in clusters])

    hitThresholdMask = (numHits>=numHitsThreshold)
    filteredPDGCodes = PDGCodes[hitThresholdMask]
    uniqueCodes = np.unique(filteredPDGCodes)

    for i, feature in enumerate(features):
        ax = axes[i]
        featureValues = np.array([cluster[feature] for cluster in clusters])[hitThresholdMask]
        for code in uniqueCodes:
            codeFeatureValues = featureValues[filteredPDGCodes==code]
            if len(codeFeatureValues)>0:
                if densityPlot:
                    binWeights = np.zeros_like(codeFeatureValues) + 1/len(codeFeatureValues)
                    ax.hist(codeFeatureValues, bins=50, label=f"PDGCode: {code}", histtype="step", lw=2, weights=binWeights)
                else:
                    ax.hist(codeFeatureValues, bins=50, label=f"PDGCode: {code}", histtype="step", lw=2)
            
        ax.set_title(f"{feature} Distribution", fontsize=8)
        ax.set_xlabel("Value", fontsize=6)
        ax.set_ylabel("Proportions" if densityPlot else "Frequency", fontsize=6)
        ax.legend()
        ax.tick_params(axis='both', labelsize=6)
    
    plt.tight_layout()
    plt.show()
    return None


def main():
    args = parser()
    dataFile, numHitsThreshold, densityPlot = args.datafile, args.numhitsthreshold, args.densityplot

    clusters = dataUnpack(dataFile)
    #features = ["linearRmsError", "transverseWidth", "meanEnergyDep", "rmsRateEnergyDeposition", "endpointsDistance", "numHits", "rmsHitGap", "edgeProximity", "angle"]
    features = ["purity", "completeness"]
    featurePlot(clusters, features, numHitsThreshold, densityPlot)

    return None

main()