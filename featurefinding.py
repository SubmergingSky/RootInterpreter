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

# Returns a list of clusters (dict) each representing the hits from a given particle
def createClusters(dataFile):
    with open(dataFile, "r") as f:
        data = json.load(f)
    
    events = []
    for evData in data:
        event = [[], np.array(evData["hitPositions"])] # Counters, hits
        for id in evData["hitIds"]: event[0].append(int(str(id)[-5:]))
        events.append(event)

    clusters = []
    for i in range(len(events)):
        counters, hits = events[i]
        uniqueCounters = np.unique(counters)
        for counter in uniqueCounters:
            cluster = {
                "eventId": i,
                "counter": int(counter),
                "hits": hits[(counters==counter)].tolist()
            }
            clusters.append(cluster)

    return clusters

# Calculates the rms of each cluster's hits to a straight line and appends this to the cluster.
def rmsLinearFit(clusters):
    for cluster in clusters:
        hits = cluster["hits"]
        if len(hits)<2:
            cluster["rmserror"] = 0
            continue

        centroid = np.mean(hits, axis=0)
        U, s, Vt = np.linalg.svd(hits-centroid)
        fittedLineDirection = Vt[0] / np.linalg.norm(Vt[0])
        projectedPoints = centroid + np.outer(np.dot((hits-centroid), fittedLineDirection), fittedLineDirection)
        perpendicularDistances = np.linalg.norm((hits-projectedPoints), axis=1)
        cluster["rmserror"] = np.sqrt(np.mean(perpendicularDistances**2))
    
    return clusters

def energyDeposition(clusters):
    return None
        
def findFeatures(clusters):
    clusters = rmsLinearFit(clusters)
    return clusters


def main():
    args = parser()
    dataFile, output = args.datafile, args.output

    unfeaturedClusters = createClusters(dataFile)
    featuredClusters = findFeatures(unfeaturedClusters)

    if output:
        with open("Data/clusters.json", "w") as f:
            json.dump(featuredClusters, f, indent=4)
        print("Output file created")


main()