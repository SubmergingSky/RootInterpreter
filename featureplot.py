import numpy as np
import pandas as pd
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

def evaluationPlot(clusters, threeD=False):
    numHits, purities, completenesses, accepts = [], [], [], []
    for c in clusters:
        numHits.append(c["numMCHits"])
        purities.append(c["purity"])
        completenesses.append(c["completeness"])
        accepts.append(c["acceptReco"])

    if threeD:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("# Hits")
        ax.set_ylabel("Purity")
        ax.set_zlabel("Completeness")
        ax.set_title('Evaluation Plot')
        ax.scatter(numHits, purities, completenesses, s=0.4)
        ax.autoscale_view()
    else:
        fig = plt.figure(figsize=(8,6))
        df = pd.DataFrame({"numHits": numHits, "accepted": accepts})
        step = 200
        bins = np.arange(0, df["numHits"].max()+step, step)
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
        df["hitRange"] = pd.cut(x=df["numHits"], bins=bins, labels=labels, right=False)

        acceptance = df.groupby("hitRange")["accepted"].mean().reset_index()
        plt.bar(acceptance["hitRange"], acceptance["accepted"], color="blue")
        plt.xlabel('Number of Hits')
        plt.ylabel("Acceptance Rate")
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--')


    plt.show()


def main():
    args = parser()
    dataFile, numHitsThreshold, densityPlot = args.datafile, args.numhitsthreshold, args.densityplot

    clusters = dataUnpack(dataFile)
    features = ["linearRmsError", "transverseWidth", "meanEnergyDep", "rmsRateEnergyDeposition", "endpointsDistance", "numHits", "rmsHitGap", "edgeProximity", "angle"]
    #featurePlot(clusters, features, numHitsThreshold, densityPlot)
    evaluationPlot(clusters)

    return None

main()