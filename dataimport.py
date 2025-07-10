import numpy as np
import uproot
import json
import argparse

def parser():
    parser = argparse.ArgumentParser(description="Process ROOT event data and save to JSON.")
    parser.add_argument(
        "-d", "--datafile",
        type=str,
        default="data.root",
        help="Path of the input ROOT data file."
    )
    parser.add_argument(
        "-o", "--outputfile",
        type=str,
        default="data.json",
        help="Path of the output JSON file."
    )
    parser.add_argument(
        "-t", "--treename",
        type=str,
        default="info",
        help="Name of the ROOT tree."
    )
    return parser.parse_args()

# Unpacks a root file and returns a tuple of event data.
def dataUnpack(filename, treename):
    with uproot.open(f"{filename}:{treename}") as tree:
        counters = tree["counters"].array(library="np")
        PDGCodes = tree["PDGCodes"].array(library="np")
        originCodes = tree["originCodes"].array(library="np")
        numMCHits, numPfoHits = tree["numMCHits"].array(library="np"), tree["numPfoHits"].array(library="np")
        purities, completenesses = tree["purities"].array(library="np"), tree["completenesses"].array(library="np")
        hitCounters = tree["hitCounters"].array(library="np")
        hitsX, hitsY, hitsZ = tree["hitsX"].array(library="np"), tree["hitsY"].array(library="np"), tree["hitsZ"].array(library="np") #[mm]
        cellSizesX, cellSizesY, cellSizesZ = tree["cellSizesX"].array(library="np"), tree["cellSizesY"].array(library="np"), tree["cellSizesZ"].array(library="np") #[mm]
        inputEnergies = tree["inputEnergies"].array(library="np")
        
    print(f"There are {len(PDGCodes)} total events.")
    return np.array((counters, PDGCodes, originCodes, numMCHits, numPfoHits, purities, completenesses, hitCounters, hitsX, hitsY, hitsZ, cellSizesX, cellSizesY, cellSizesZ, inputEnergies)).T

# Packages the event data into clusters linking together hits from the same particle.
def createClusters(data):
    clusters = []

    for i, eventData in enumerate(data):
        counters, PDGCodes, originCodes, numMCHits, numPfoHits, purities, completenesses, hitCounters, hitsX, hitsY, hitsZ, cellSizesX, cellSizesY, cellSizesZ, inputEnergies = eventData
        eventHits, eventHitGeometries = np.column_stack((hitsX, hitsY, hitsZ)), np.column_stack((cellSizesX, cellSizesY, cellSizesZ))

        for k, c in enumerate(counters):
            cluster = {
                "eventId": i,
                "counter": int(c),
                "PDGCode": int(PDGCodes[k]),
                "isClearCosmic": bool(originCodes[k]==7),
                "isFromNeutrino": bool(originCodes[k]==8),
                "numMCHits": int(numMCHits[k]),
                "numPfoHits": int(numPfoHits[k]),
                "purity": float(purities[k]),
                "completeness": float(completenesses[k]),
                "hits": eventHits[(hitCounters==c)].tolist(),
                "hitGeometries":eventHitGeometries[(hitCounters==c)].tolist(),
                "inputEnergies": inputEnergies[(hitCounters==c)].tolist()
            }
            clusters.append(cluster)

    return clusters

# Cleans the clusters by vetoing those that fulfill certain conditions.
def cleanClusters(clusters, onlyNeutrino=False):
    if onlyNeutrino:
        clusters = [cluster for cluster in clusters if cluster["isFromNeutrino"]==True]
        clusters = [cluster for cluster in clusters if cluster["PDGCode"]!=22]

    return clusters

# Outputs the clusters to a json file.
def dataOutput(data, outputFile):
    with open(f"Data/{outputFile}", "w") as f:
        json.dump(data, f, indent=4)
    print("Output file created")
    return None


def main():
    args = parser()
    dataFile, treename, outputFile = args.datafile, args.treename, args.outputfile

    data = dataUnpack(f"Data/{dataFile}", treename)
    clusters = createClusters(data)
    print(f"There are {len(clusters)} total uncleaned clusters.")
    clusters = cleanClusters(clusters)
    print(f"There are {len(clusters)} total cleaned clusters.")
    dataOutput(clusters, outputFile)

    return None

main()