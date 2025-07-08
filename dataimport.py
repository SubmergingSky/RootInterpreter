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
        ids = tree["ids"].array(library="np")
        hitIds = tree["hitIds"].array(library="np") # 8/9 nnnnnnnnnn "PDG code" nnnnn "particle counter"
        hitsX, hitsY, hitsZ = tree["hitsX"].array(library="np"), tree["hitsY"].array(library="np"), tree["hitsZ"].array(library="np") #[mm]
        cellSizesX, cellSizesY, cellSizesZ = tree["cellSizesX"].array(library="np"), tree["cellSizesY"].array(library="np"), tree["cellSizesZ"].array(library="np") #[mm]
        inputEnergies = tree["inputEnergies"].array(library="np")

    print(f"There are {len(ids)} total events.")
    return (ids, hitIds, hitsX, hitsY, hitsZ, cellSizesX, cellSizesY, cellSizesZ, inputEnergies)

# Packages the event data into clusters linking together hits from the same particle.
def createClusters(data):
    clusters = []
    ids, hitIds, hitsX, hitsY, hitsZ, cellSizesX, cellSizesY, cellSizesZ, inputEnergies = data
    for i in range(ids.shape[0]):
        eventHits, eventHitGeometries = np.column_stack((hitsX[i], hitsY[i], hitsZ[i])), np.column_stack((cellSizesX[i], cellSizesY[i], cellSizesZ[i]))

        uniqueHitIds = np.unique(hitIds[i])
        for hitId in uniqueHitIds:
            cluster = {
                "eventId": i,
                "hitId": int(hitId),
                "isFromNeutrino": str(hitId)[0]=="8",
                "isClearCosmic": str(hitId)[0]=="7",
                "PDGCode": int(str(hitId)[1:11]),
                "counter": int(str(hitId)[-5:]),
                "hits": eventHits[(hitIds[i]==hitId)].tolist(),
                "hitGeometries": eventHitGeometries[(hitIds[i]==hitId)].tolist(),
                "inputEnergies": inputEnergies[i][(hitIds[i]==hitId)].tolist()
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