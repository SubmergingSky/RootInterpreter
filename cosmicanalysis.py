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

    cosmicClusters, ambiguousClusters = [c for c in selectedClusters if c["truthOrigin"]==7], [c for c in selectedClusters if c["truthOrigin"]!=7]
    print(f"Clear cosmic clusters: {len(cosmicClusters)}     Ambiguous clusters: {len(ambiguousClusters)}\n")

    return (cosmicClusters, ambiguousClusters)

def efficiencyPlot(combinedClusters):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for i, clusters in enumerate(combinedClusters):
        ax = axes[i]
        numHits, accepts, origins = [], [], []
        for c in clusters:
            numHits.append(c["numMCHits"])
            accepts.append(c["acceptReco"])
            accepts.append(c["recoOrigin"])
        
        df = pd.DataFrame({"numHits": numHits, "accepted": accepts})
        binsLow, binsMid, binsHigh = np.arange(0, 100, 100), np.arange(100, 2000, 100), np.arange(2000, np.max(numHits)+500, 100)
        bins = np.concatenate((binsLow, binsMid, binsHigh))
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
        df["ranges"] = pd.cut(x=df["numHits"], bins=bins, labels=labels, right=False)

        acceptance = df.groupby("ranges", observed=True)["accepted"].mean().reset_index()
        ax.bar(acceptance["ranges"], acceptance["accepted"], color="blue")

        labels = acceptance["ranges"].tolist()
        reducedLabels = [label if i%3==0 else "" for i,label in enumerate(labels)]
        ax.set_xticks(ticks=range(len(labels)), labels=reducedLabels, rotation=45, ha="right")

        
        ax.set_xlabel("# MCHits")
        ax.set_ylabel("Efficiency")
        ax.set_ylim(0,1)
    
    axes[0].set_title("Cosmic Efficiency")
    axes[1].set_title("Neutrino Efficiency")
    axes[2].set_title("Ambiguous Efficiency")
    plt.tight_layout()
    plt.show()
    return None

def cosmicPlot(clusters):
    plt.figure(figsize=(10,6))

    numHits, accepts, originCodes = [], [], []
    for c in clusters:
        numHits.append(c["numMCHits"])
        accepts.append(c["acceptReco"])
        originCodes.append(c["recoOrigin"])
    recoCosmics, recoAmbiguous = [n for i,n in enumerate(numHits) if originCodes[i]==7], [n for i,n in enumerate(numHits) if originCodes[i]!=7]
    combinedData = [recoCosmics, recoAmbiguous]
    labels, colours = ["Reco Cosmics", "Reco Ambiguous"], ["lightblue", "gold"]
    bins = np.arange(0, np.max(numHits)+101, 100)

    plt.hist(combinedData, bins=bins, stacked=True, label=labels, color=colours)

    plt.xlabel("# MCHits")
    plt.ylabel("Number of Clusters")
    plt.legend()
    plt.tight_layout()
    #plt.show()

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    
    df = pd.DataFrame({"numHits": numHits, "accepted": accepts, "origin": originCodes})

    conditions = [df["origin"]==7, df["origin"].isin([6,8,9])]
    choices = ["Clear Cosmic", "Ambiguous"]
    df["category"] = np.select(conditions, choices, default="Ambiguous")
    colours = {"Clear Cosmic": "lightblue", "Ambiguous": "gold"}
    




    bins = np.arange(0, np.max(numHits)+101, 100)
    labels = [f"[{bins[i]}-{bins[i+1]})" for i in range(len(bins)-1)]
    df["ranges"] = pd.cut(x=df["numHits"], bins=bins, labels=labels, right=False, include_lowest=True)
    originEfficiencies = df.groupby(["ranges", "category"], observed=True)["accepted"].mean().unstack(fill_value=0)
    uniqueRanges = originEfficiencies.index.tolist()
    xIndicies = np.arange(len(uniqueRanges))

    numOrigins = len(choices)
    barWidth = 0.8/numOrigins

    for i, origin in enumerate(choices):
        offset = (i-(numOrigins-1)/2)*barWidth
        originEfficiency = originEfficiencies[origin] if origin in originEfficiencies.columns else pd.Series(0, index=uniqueRanges)
        ax.bar(xIndicies + offset, originEfficiency.to_numpy(), width=barWidth, label=f"Reco {origin}", color=colours[origin])

    ax.set_xticks(xIndicies)
    reducedLabels = [label if i%3==0 else "" for i, label in enumerate(uniqueRanges)]
    ax.set_xticklabels(reducedLabels, rotation=45, ha="right")
    ax.set_xlabel("# MCHits")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(0,1)
    ax.legend(title="Origin") 
    ax.grid(axis='y', linestyle='--', alpha=0.7) 
    ax.set_title("Efficiency by MCHits Range and Origin")

    plt.tight_layout()
    plt.show()
    return None

def categorisationQuality(combinedClusters):
    cosmicClusters, ambiguousClusters = combinedClusters
    cosmicHits, ambiguousHits = sum([c["numPfoHits"] for c in cosmicClusters]), sum([c["numPfoHits"] for c in ambiguousClusters])

    cosmicAccuracyCluster, ambiguousAccuracyCluster = sum([c["recoOrigin"]==7 for c in cosmicClusters])/len(cosmicClusters), sum([c["recoOrigin"]!=7 for c in ambiguousClusters])/len(ambiguousClusters)
    cosmicAccuracyHit, ambiguousAccuracyHit = sum([(c["recoOrigin"]==7)*c["numPfoHits"] for c in cosmicClusters])/cosmicHits, sum([(c["recoOrigin"]!=7)*c["numPfoHits"] for c in ambiguousClusters])/ambiguousHits

    cosmicEfficiencyCluster, ambiguousEfficiencyCluster = sum(c["acceptReco"] for c in cosmicClusters)/len(cosmicClusters), sum(c["acceptReco"] for c in ambiguousClusters)/len(ambiguousClusters)
    cosmicEfficiencyHit, ambiguousEfficiencyHit = sum(c["acceptReco"]*c["numPfoHits"] for c in cosmicClusters)/cosmicHits, sum(c["acceptReco"]*c["numPfoHits"] for c in ambiguousClusters)/ambiguousHits

    print(f"\n{"------ Quality Evaluation ------":^}")

    print(f"{"#":10} {"Cluster":>10} {"Hit":>10}")
    print(f"{"Cosmic":10} {len(cosmicClusters):10} {cosmicHits:10}")
    print(f"{"Ambiguous":10} {len(ambiguousClusters):10} {ambiguousHits:10}\n")

    print(f"{"Accuracy:":10}")
    print(f"{"Cosmic":10} {cosmicAccuracyCluster:10.4f} {cosmicAccuracyHit:10.4f}")
    print(f"{"Ambiguous":10} {ambiguousAccuracyCluster:10.4f} {ambiguousAccuracyHit:10.4f}\n")

    print("Efficiency:")
    print(f"{"Cosmic":10} {cosmicEfficiencyCluster:10.4f} {cosmicEfficiencyHit:10.4f}")
    print(f"{"Ambiguous":10} {ambiguousEfficiencyCluster:10.4f} {ambiguousEfficiencyHit:10.4f}\n")

    return None



def main():
    args = parser()
    dataFile, numHitsThreshold, densityPlot = args.datafile, args.numhitsthreshold, args.densityplot

    clusters = dataUnpack(dataFile, numHitsThreshold)
    #categorisationQuality(clusters)
    #efficiencyPlot(clusters)
    cosmicPlot(clusters[0])

    return None

main()