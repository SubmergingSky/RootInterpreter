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
        uids = tree["uids"].array(library="np")
        truthPDGs, recoPDGs = tree["truthPDGs"].array(library="np"), tree["recoPDGs"].array(library="np")
        truthOrigins, recoOrigins = tree["truthOrigins"].array(library="np"), tree["recoOrigins"].array(library="np")
        numPfoHits = tree["numPfoHits"].array(library="np")
        purities, completenesses = tree["purities"].array(library="np"), tree["completenesses"].array(library="np")
        hitCounters = tree["hitCounters"].array(library="np")
        hitsX, hitsY, hitsZ = tree["hitsX"].array(library="np"), tree["hitsY"].array(library="np"), tree["hitsZ"].array(library="np") #[mm]
        cellSizesX, cellSizesY, cellSizesZ = tree["cellSizesX"].array(library="np"), tree["cellSizesY"].array(library="np"), tree["cellSizesZ"].array(library="np") #[mm]
        inputEnergies = tree["inputEnergies"].array(library="np")
        numMCHitsU, numMCHitsV, numMCHitsW = tree["numMCHitsU"].array(library="np"), tree["numMCHitsV"].array(library="np"), tree["numMCHitsW"].array(library="np")
        MCHitCountersU, MCHitCountersV, MCHitCountersW = tree["MCHitCountersU"].array(library="np"), tree["MCHitCountersV"].array(library="np"), tree["MCHitCountersW"].array(library="np")
        MCHitsUX, MCHitsUZ, MCHitsVX, MCHitsVZ, MCHitsWX, MCHitsWZ = tree["MCHitsUX"].array(library="np"), tree["MCHitsUZ"].array(library="np"), tree["MCHitsVX"].array(library="np"), tree["MCHitsVZ"].array(library="np"), tree["MCHitsWX"].array(library="np"), tree["MCHitsWZ"].array(library="np")
        
    print(f"There are {len(counters)} total events.")
    return np.array((counters, uids, truthPDGs, recoPDGs, truthOrigins, recoOrigins, numPfoHits, purities, completenesses, hitCounters, hitsX, hitsY, hitsZ, cellSizesX, cellSizesY, cellSizesZ, inputEnergies, numMCHitsU, numMCHitsV, numMCHitsW, MCHitCountersU, MCHitCountersV, MCHitCountersW, MCHitsUX, MCHitsUZ, MCHitsVX, MCHitsVZ, MCHitsWX, MCHitsWZ)).T

# Packages the event data into clusters linking together hits from the same particle.
def createClusters(data):
    clusters = []

    for i, eventData in enumerate(data):
        counters, uids, truthPDGs, recoPDGs, truthOrigins, recoOrigins, numPfoHits, purities, completenesses, hitCounters, hitsX, hitsY, hitsZ, cellSizesX, cellSizesY, cellSizesZ, inputEnergies, numMCHitsU, numMCHitsV, numMCHitsW, MCHitCountersU, MCHitCountersV, MCHitCountersW, MCHitsUX, MCHitsUZ, MCHitsVX, MCHitsVZ, MCHitsWX, MCHitsWZ = eventData
        eventHits, eventHitGeometries = np.column_stack((hitsX, hitsY, hitsZ)), np.column_stack((cellSizesX, cellSizesY, cellSizesZ))
        MCHitsU, MCHitsV, MCHitsW = np.column_stack((MCHitsUX, MCHitsUZ)), np.column_stack((MCHitsVX, MCHitsVZ)), np.column_stack((MCHitsWX, MCHitsWZ))

        for k, c in enumerate(counters):
            cluster = {
                "include": True,
                "eventId": i,
                "counter": int(c),
                "MCUid": int(uids[k]),
                "truthPDG": int(truthPDGs[k]),
                "recoPDG": int(recoPDGs[k]),
                "truthOrigin": int(truthOrigins[k]),
                "recoOrigin": int(recoOrigins[k]),
                "numPfoHits": int(numPfoHits[k]),
                "purity": float(purities[k]),
                "completeness": float(completenesses[k]),
                #"hits": eventHits[(hitCounters==c)].tolist(),
                #"hitGeometries":eventHitGeometries[(hitCounters==c)].tolist(),
                #"inputEnergies": inputEnergies[(hitCounters==c)].tolist(),
                "numMCHits": [int(numMCHitsU[k]), int(numMCHitsV[k]), int(numMCHitsW[k])],
                #"MCHitsU": MCHitsU[(MCHitCountersU==c)].tolist(),
                #"MCHitsV": MCHitsV[(MCHitCountersV==c)].tolist(),
                "MCHitsW": MCHitsW[(MCHitCountersW==c)].tolist()
            }
            clusters.append(cluster)

    return clusters

# Cleans the clusters by vetoing those that fulfill certain conditions.
def cleanClusters(clusters, onlyNeutrino=False):
    if onlyNeutrino:
        clusters = [c for c in clusters if c["isFromNeutrino"]==True]
        clusters = [c for c in clusters if c["recoPDG"]!=22]

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