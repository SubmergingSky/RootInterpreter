import numpy as np
import uproot
import json
import argparse

def parser():
    parser = argparse.ArgumentParser(description="Process ROOT event data and save to JSON.")
    parser.add_argument(
        "-d", "--datafile",
        type=str,
        default="Data/data.root",
        help="Path to the input ROOT data file."
    )
    parser.add_argument(
        "-o", "--outputfile",
        type=str,
        default="Data/data.json",
        help="Name of the output JSON file."
    )
    parser.add_argument(
        "-t", "--treename",
        type=str,
        default="mc_info",
        help="Name of the ROOT tree."
    )
    return parser.parse_args()

# Unpacks a root file and returns a tuple of event data
def dataUnpack(filename, treename):
    with uproot.open(f"{filename}:{treename}") as tree:
        ids = tree["ids"].array(library="np")
        energies = tree["energies"].array(library="np") #[GeV]
        hitIds = tree["hitIds"].array(library="np") # 8/9 nnnnnnnnnn "PDG code" nnnnn "particle counter"
        hitsX, hitsZ = tree["hitsX"].array(library="np"), tree["hitsZ"].array(library="np") #[mm]
        inputEnergies = tree["inputEnergies"].array(library="np")

    return (ids, energies, hitIds, hitsX, hitsZ, inputEnergies)

# Packages the event data into clusters linking together hits from the same particle
def createClusters(data):
    clusters = []
    ids, energies, hitIds, hitsX, hitsZ, inputEnergies = data
    for i in range(ids.shape[0]):
        eventHits = np.column_stack((hitsX[i], hitsZ[i]))

        uniqueHitIds = np.unique(hitIds[i])
        for hitId in uniqueHitIds:
            cluster = {
                "eventId": i,
                "hitId": int(hitId),
                "isFromNeutrino": str(hitId)[0]=="8",
                "PDGCode": int(str(hitId)[1:11]),
                "counter": int(str(hitId)[-5:]),
                "hits": eventHits[(hitIds[i]==hitId)].tolist(),
                "inputEnergies": inputEnergies[i][(hitIds[i]==hitId)].tolist()
            }
            clusters.append(cluster)

    return clusters

def dataOutput(data, outputFile):
    with open(outputFile, "w") as f:
        json.dump(data, f, indent=4)
    print("Output file created")
    return None


def main():
    args = parser()
    dataFile, treename, outputFile = args.datafile, args.treename, args.outputfile

    data = dataUnpack(dataFile, treename)
    clusters = createClusters(data)
    dataOutput(clusters, outputFile)

    return None

main()