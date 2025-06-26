import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

def parser():
    parser = argparse.ArgumentParser(description="Uses hit positions to identify features.")
    parser.add_argument(
        "-d", "--datafile",
        type=str,
        default="Data/data.json",
        help="Path to the input ROOT data file."
    )
    parser.add_argument(
        "-o", "--output",
        action="store_true",
        default=False,
        help="Whether to output the clusters to a json file."
    )
    parser.add_argument(
        "-t", "--numhitsthreshold",
        type=int,
        default=4,
        help="The minimum number of hits a cluster must have to be considered."
    )
    parser.add_argument(
        "-p", "--onlypm",
        action="store_true",
        default=True,
        help="Whether to only show pions and muons in the result plot."
    )
    return parser.parse_args()

# Calculates the rms of each cluster's hits to a straight line and appends this to the cluster.
def rmsLinearFit(cluster):
    hits = cluster["hits"]
    if len(hits)<2:
        linearRmsError = 0
    else:
        centroid = np.mean(hits, axis=0)
        U, s, Vt = np.linalg.svd(hits-centroid)
        fittedLineDirection = Vt[0] / np.linalg.norm(Vt[0])
        projectedPoints = centroid + np.outer(np.dot((hits-centroid), fittedLineDirection), fittedLineDirection)
        perpendicularDistances = np.linalg.norm((hits-projectedPoints), axis=1)
        linearRmsError = np.sqrt(np.mean(perpendicularDistances**2))
    
    cluster["linearRmsError"] = linearRmsError
    return cluster

# Calculates the average rate of energy deposition and appends this to the cluster.
def meanEnergyDeposition(cluster):
    inputEnergies = cluster["inputEnergies"]
    cluster["meanEnergyDeposition"] = np.mean(inputEnergies)
    return cluster

# Plots a histogram of RMS error for each particle type
def rmsErrorPlot(clusters, numHitsthreshold, onlypm):
    numHits = np.array([len(cluster.get("hits")) for cluster in clusters])
    PDGCodes, rmsValues= np.array([cluster.get("PDGCode") for cluster in clusters])[numHits>=numHitsthreshold], np.array([cluster.get("linearRmsError") for cluster in clusters])[numHits>=numHitsthreshold], 
    for code in np.unique(PDGCodes):
        if onlypm and (code!=211 and code !=13): continue
        codeRmsValues = rmsValues[PDGCodes==int(code)]
        if len(codeRmsValues)>0: plt.hist(codeRmsValues, bins=50, range=(0, 20), label=f"PDGCode: {code}")
    
    plt.xlabel("RMS Values")
    plt.ylabel("# Results")
    plt.legend()
    plt.show()
    return None

def findFeatures(clusters):
    for cluster in clusters:
        cluster = rmsLinearFit(cluster)
        cluster = meanEnergyDeposition(cluster)

    return clusters

def main():
    args = parser()
    dataFile, output, numHitsThreshold, onlypm = args.datafile, args.output, args.numhitsthreshold, args.onlypm

    with open(dataFile, "r") as f:
        data = json.load(f)
    
    featuredClusters = findFeatures(data)
    print(f"There are {len(featuredClusters)} total clusters.")
    rmsErrorPlot(featuredClusters, numHitsThreshold, onlypm)

    if output:
        with open("Data/featured_data.json", "w") as f:
            json.dump(featuredClusters, f, indent=4)
        print("Output file created")

    return None

main()