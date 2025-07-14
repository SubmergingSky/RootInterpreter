import numpy as np
import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt

def parser():
    parser = argparse.ArgumentParser(description="Analysis of cosmic ray tagging.")
    parser.add_argument(
        "-d", "--datafile",
        type=str,
        default="featured_data1_10.json",
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

def dataUnpack(dataFile, numHitsThreshold):
    with open(f"Data/{dataFile}", "r") as f:
        data = json.load(f)
    
    fullClusters = np.array([c for c in data])
    selectedClusters = [c for c in fullClusters if c["numPfoHits"]>=numHitsThreshold]
    print(f"Total clusters: {len(fullClusters)}     Selected clusters: {len(selectedClusters)}")

    cosmicClusters, neutrinoClusters = [c for c in selectedClusters if c["isClearCosmic"]], [c for c in selectedClusters if c["isFromNeutrino"]]
    print(f"Cosmic clusters: {len(cosmicClusters)}     Neutrino clusters: {len(neutrinoClusters)}\n")

    return (cosmicClusters, neutrinoClusters)

def efficiencyPlot(combinedClusters):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axStep = 100
    for i, clusters in enumerate(combinedClusters):
        ax = axes[i]
        numHits, accepts = [], []
        for c in clusters:
            numHits.append(c["numMCHits"])
            accepts.append(c["acceptReco"])
        
        df = pd.DataFrame({"numHits": numHits, "accepted": accepts})
        bins = np.arange(0, np.max(numHits)+axStep, axStep)
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
        df["ranges"] = pd.cut(x=df["numHits"], bins=bins, labels=labels, right=False)

        acceptance = df.groupby("ranges")["accepted"].mean().reset_index()
        ax.bar(acceptance["ranges"], acceptance["accepted"], color="blue")

        labels = acceptance["ranges"].tolist()
        reducedLabels = [label if i%3==0 else "" for i,label in enumerate(labels)]
        ax.set_xticks(ticks=range(len(labels)), labels=reducedLabels, rotation=45, ha="right")

        
        ax.set_xlabel("# MCHits")
        ax.set_ylabel("Efficiency")
        ax.set_ylim(0,1)
    
    axes[0].set_title("Clear Cosmic Efficiency")
    axes[1].set_title("Neutrino Efficiency")
    plt.tight_layout()
    plt.show()
    return None








def main():
    args = parser()
    dataFile, numHitsThreshold, densityPlot = args.datafile, args.numhitsthreshold, args.densityplot

    clusters = dataUnpack(dataFile, numHitsThreshold)
    efficiencyPlot(clusters)

    return None

main()