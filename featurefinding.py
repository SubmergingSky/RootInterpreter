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
        "-m", "--onlypimu",
        action="store_true",
        default=True,
        help="Whether to only show pions and muons in the result plot."
    )
    parser.add_argument(
        "-p", "--densityplot",
        action="store_true",
        default=False,
        help="Whether to plot event densities rather than absolute quantities."
    )
    return parser.parse_args()

# Calculates the rms of each cluster's hits to a straight line.
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

# Calculates the average rate of energy deposition.
def meanEnergyDeposition(cluster):
    inputEnergies = cluster["inputEnergies"]
    cluster["meanEnergyDeposition"] = np.mean(inputEnergies)
    return cluster

# Calculates the distance between the track's endpoints in mm.
def endPointsDistance(cluster):
    trackStart, trackEnd = np.array(cluster["hits"][0]), np.array(cluster["hits"][-1])
    cluster["trackLength"] = np.linalg.norm(trackEnd-trackStart)
    return cluster

# Calculates the number of hits in the particle's track.
def numHits(cluster):
    cluster["numHits"] = len(cluster["hits"])
    return cluster

# Calculates each feature and appends to each cluster.
def findFeatures(clusters):
    for cluster in clusters:
        cluster = rmsLinearFit(cluster)
        cluster = meanEnergyDeposition(cluster)
        cluster = endPointsDistance(cluster)
        cluster = numHits(cluster)

    return clusters


# Plots a histogram of a given cluster feature.
def featurePlot(clusters, feature, numHitsThreshold, onlyPiMu, densityPlot):
    numHits = np.array([len(cluster.get("hits")) for cluster in clusters])
    PDGCodes, featureValues= np.array([cluster.get("PDGCode") for cluster in clusters])[numHits>=numHitsThreshold], np.array([cluster.get(feature) for cluster in clusters])[numHits>=numHitsThreshold], 
    for code in np.unique(PDGCodes):
        if onlyPiMu and (code!=211 and code !=13):
            continue
        else:
            codeFeatureValues = featureValues[PDGCodes==int(code)]
            if len(codeFeatureValues)>0:
                if densityPlot:
                    binWeights = np.zeros_like(codeFeatureValues) + 1/len(codeFeatureValues)
                    plt.hist(codeFeatureValues, bins=50, label=f"PDGCode: {code}", histtype="step", lw=2, weights=binWeights)
                else:
                    plt.hist(codeFeatureValues, bins=50, label=f"PDGCode: {code}", histtype="step", lw=2)
            else:
                continue
    
    plt.xlabel(f"{feature} Values")
    if densityPlot:
        plt.ylabel("Result Proportions")
    else:
        plt.ylabel("# Results")
    plt.legend()
    plt.title(f"{feature} Values")
    plt.show()
    return None

def main(feature="linearRmsError"):
    args = parser()
    dataFile, output, numHitsThreshold, onlyPiMu, densityPlot = args.datafile, args.output, args.numhitsthreshold, args.onlypimu, args.densityplot

    with open(dataFile, "r") as f:
        data = json.load(f)
    
    featuredClusters = findFeatures(data)
    print(f"There are {len(featuredClusters)} total clusters.")
    featurePlot(featuredClusters, feature, numHitsThreshold, onlyPiMu, densityPlot)

    if output:
        with open("Data/featured_data.json", "w") as f:
            json.dump(featuredClusters, f, indent=4)
        print("Output file created")

    return None

main()