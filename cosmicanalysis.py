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
        default=10,
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
    selectedClusters = [c for c in fullClusters if (c["numPfoHits"]>=numHitsThreshold and c["include"])]
    print(f"Total clusters: {len(fullClusters)}     Selected clusters: {len(selectedClusters)}")

    cosmicClusters, neutrinoClusters, ambiguousClusters = [c for c in selectedClusters if c["truthOrigin"]==7], [c for c in selectedClusters if c["truthOrigin"]==8], [c for c in selectedClusters if c["truthOrigin"] not in [7,8]]
    print(f"Clear cosmic clusters: {len(cosmicClusters)}     Neutrino clusters: {len(neutrinoClusters)}     Ambiguous clusters: {len(ambiguousClusters)}\n")

    return (cosmicClusters, neutrinoClusters, ambiguousClusters)

def efficiencyPlot(combinedClusters):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for i, clusters in enumerate(combinedClusters):
        ax = axes[i]
        numHits, accepts, origins = [], [], []
        for c in clusters:
            numHits.append(c["numMCHits"])
            accepts.append(c["acceptReco"])
            origins.append(c["recoOrigin"])
        
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
    cosmicClusters, neutrinoClusters, ambiguousClusters = combinedClusters
    combinedClustersMasked = [(cosmicClusters, [7]), (neutrinoClusters, [8]), (ambiguousClusters, [6,9])]
    hits, accuracyCluster, accuracyHit, efficiencyCluster, efficiencyHit = [], [], [], [], []
    for (clusters, mask) in combinedClustersMasked:
        cHits = sum([c["numPfoHits"] for c in clusters])

        hits.append(cHits)
        accuracyCluster.append( sum([c["recoOrigin"] in mask for c in clusters])/len(clusters) )
        accuracyHit.append( sum([(c["recoOrigin"] in mask)*c["numPfoHits"] for c in clusters])/cHits )
        efficiencyCluster.append( sum(c["acceptReco"] for c in clusters)/len(clusters) )
        efficiencyHit.append( sum(c["acceptReco"]*c["numPfoHits"] for c in clusters)/cHits )

    trueNeutrinoHits, markedNeutrinoHits = [], []
    allClusters = cosmicClusters + neutrinoClusters + ambiguousClusters
    for c in allClusters:
        allMCHits = c["MCHitsU"] + c["MCHitsV"] + c["MCHitsW"]
        if c["truthReco"]==8:
            trueNeutrinoHits.extend(allMCHits)



    print(f"\n{"------ Quality Evaluation ------":^40}")

    print(f"{"#":20} {"Cluster":>10} {"Hit":>10}")
    print(f"{"Clear Cosmic":20} {len(cosmicClusters):10} {hits[0]:10}")
    print(f"{"Ambiguous (Neutrino)":20} {len(neutrinoClusters):10} {hits[1]:10}")
    print(f"{"Ambiguous (Other)":20} {len(ambiguousClusters):10} {hits[2]:10}\n")

    print(f"{"Accuracy:":20}")
    print(f"{"Clear Cosmic":20} {accuracyCluster[0]:10.4f} {accuracyHit[0]:10.4f}")
    print(f"{"Ambiguous (Neutrino)":20} {accuracyCluster[1]:10.4f} {accuracyHit[1]:10.4f}")
    print(f"{"Ambiguous (Other)":20} {accuracyCluster[2]:10.4f} {accuracyHit[2]:10.4f}\n")

    print("Efficiency:")
    print(f"{"Clear Cosmic":20} {efficiencyCluster[0]:10.4f} {efficiencyHit[0]:10.4f}")
    print(f"{"Ambiguous (Neutrino)":20} {efficiencyCluster[1]:10.4f} {efficiencyHit[1]:10.4f}")
    print(f"{"Ambiguous (Other)":20} {efficiencyCluster[2]:10.4f} {efficiencyHit[2]:10.4f}\n")

    #prop = sum([c["truthOrigin"]==7 for c in cosmicClusters+neutrinoClusters+ambiguousClusters])/sum([(c["truthOrigin"]==6 or c["truthOrigin"]==7) for c in cosmicClusters+neutrinoClusters+ambiguousClusters])
    #print(prop)

    return None



def main():
    args = parser()
    dataFile, numHitsThreshold, densityPlot = args.datafile, args.numhitsthreshold, args.densityplot

    clusters = dataUnpack(dataFile, numHitsThreshold)
    categorisationQuality(clusters)
    #efficiencyPlot(clusters)
    #cosmicPlot(clusters[0])

    return None

main()