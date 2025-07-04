import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from sklearn.decomposition import PCA

def parser():
    parser = argparse.ArgumentParser(description="Uses hit positions to identify features.")
    parser.add_argument(
        "-d", "--datafile",
        type=str,
        default="data.json",
        help="Path to the input data file."
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
        default=False,
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
    if len(hits)<3:
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

def transverseWidth(cluster):
    hits = cluster["hits"]
    if len(hits)<2:
        width = 0
    else:
        pca = PCA(n_components=2)
        pca.fit(hits)
        width = np.sqrt(pca.explained_variance_[1])
    
    cluster["transverseWidth"] = width
    return cluster

# Calculates the average rate of energy deposition.
def meanEnergyDeposition(cluster):
    inputEnergies = cluster["inputEnergies"]
    cluster["meanEnergyDeposition"] = np.mean(inputEnergies)
    return cluster

# Calculates the mean RMS difference in energy deposition between hits.
def rmsRateEnergyDeposition(cluster):
    inputEnergies = cluster["inputEnergies"]
    if len(inputEnergies)<2:
        rmsRate = 0
    else:
        differences = np.diff(inputEnergies)
        rmsRate = np.sqrt(np.mean(differences**2))
    cluster["rmsRateEnergyDeposition"] = rmsRate
    return cluster

# Calculates the distance between the track's endpoints in mm.
def endPointsDistance(cluster):
    trackStart, trackEnd = np.array(cluster["hits"][0]), np.array(cluster["hits"][-1])
    cluster["endpointsDistance"] = np.linalg.norm(trackEnd-trackStart)
    return cluster

# Calculates the number of hits in the particle's track.
def numHits(cluster):
    cluster["numHits"] = len(cluster["hits"])
    return cluster

# Calculates of the mean RMS gap between hits.
def rmsHitGap(cluster):
    hits, geometries = cluster["hits"], cluster["hitGeometries"]
    if len(hits)<2:
        rmsGap = 0
    else:
        gapLengths = []
        for i in range(len(hits)-1):
            x1, z1, x2, z2 = hits[i][0], hits[i][1], hits[i+1][0], hits[i+1][1]
            w1, h1, w2, h2 = geometries[i][0], geometries[i][1], geometries[i+1][0], geometries[i+1][1]
            xGap, zGap = max(0, abs(x2-x1)-(w1+w2)/2), max(0, abs(z2-z1)-(h1+h2)/2)
            gapLengths.append(np.sqrt(xGap**2+zGap**2))
        rmsGap = np.sqrt(np.mean(np.array(gapLengths)**2))
    
    cluster["rmsHitGap"] = rmsGap
    return cluster

# Calculates the closest proximity to the edge of the detector.
def edgeProximity(cluster):
    xMin, xMax, yMin, yMax = -203, 203, 0, 505
    hits = np.array(cluster["hits"])
    leftDist, rightDist, topDist, bottomDist = hits[:,0] - xMin, xMax - hits[:,0], yMax - hits[:,1], hits[:,1] - yMin
    combinedDist = np.stack((leftDist, rightDist, topDist, bottomDist), axis=1)
    minDist = np.min(combinedDist, axis=1)

    cluster["edgeProximity"] = np.min(minDist)
    return cluster

# Calculates each feature and appends to each cluster.
def findFeatures(clusters):
    for cluster in clusters:
        cluster = numHits(cluster)
        cluster = endPointsDistance(cluster)
        cluster = rmsLinearFit(cluster)
        cluster = meanEnergyDeposition(cluster)
        cluster = rmsRateEnergyDeposition(cluster)
        cluster = rmsHitGap(cluster)
        cluster = transverseWidth(cluster)
        cluster = edgeProximity(cluster)
        
    return clusters

# Removes certain keys from each cluser.
def removeKeys(clusters):
    for cluster in clusters:
        del cluster["hits"]
        del cluster["hitGeometries"]
        del cluster["inputEnergies"]

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

# Features: linearRmsError, transverseWidth, meanEnergyDeposition, rmsRateEnergyDeposition, endpointsDistance, numHits, rmsHitGap
def main(feature="rmsHitGap"):
    args = parser()
    dataFile, output, numHitsThreshold, onlyPiMu, densityPlot = args.datafile, args.output, args.numhitsthreshold, args.onlypimu, args.densityplot

    with open(f"Data/{dataFile}", "r") as f:
        data = json.load(f)
    
    featuredClusters = findFeatures(data)
    print(f"There are {len(featuredClusters)} total clusters.")
    #featurePlot(featuredClusters, feature, numHitsThreshold, onlyPiMu, densityPlot)

    if output:
        reducedClusters = removeKeys(featuredClusters)
        with open(f"Data/featured_{dataFile}", "w") as f:
            json.dump(reducedClusters, f, indent=4)
        print("Output file created")

    return None

main()