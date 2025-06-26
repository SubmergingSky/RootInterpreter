import numpy as np
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
    return parser.parse_args()

# Calculates the rms of each cluster's hits to a straight line and appends this to the cluster.
def rmsLinearFit(cluster):
    hits = cluster["hits"]
    if len(hits)<2:
        cluster["rmserror"] = 0
        return cluster

    centroid = np.mean(hits, axis=0)
    U, s, Vt = np.linalg.svd(hits-centroid)
    fittedLineDirection = Vt[0] / np.linalg.norm(Vt[0])
    projectedPoints = centroid + np.outer(np.dot((hits-centroid), fittedLineDirection), fittedLineDirection)
    perpendicularDistances = np.linalg.norm((hits-projectedPoints), axis=1)
    cluster["linearRmsError"] = np.sqrt(np.mean(perpendicularDistances**2))

    return cluster

# Calculates the average rate of energy deposition and appends this to the cluster.
def meanEnergyDeposition(cluster):
    inputEnergies = cluster["inputEnergies"]
    cluster["meanEnergyDeposition"] = np.mean(inputEnergies)
    return cluster

   
def findFeatures(clusters):
    for cluster in clusters:
        cluster = rmsLinearFit(cluster)
        cluster = meanEnergyDeposition(cluster)

    return clusters

def main():
    args = parser()
    dataFile, output = args.datafile, args.output

    with open(dataFile, "r") as f:
        data = json.load(f)
    
    featuredClusters = findFeatures(data)

    if output:
        with open("Data/featured_data.json", "w") as f:
            json.dump(featuredClusters, f, indent=4)
        print("Output file created")

    return None

main()