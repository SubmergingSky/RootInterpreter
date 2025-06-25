import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

def parser():
    parser = argparse.ArgumentParser(description="Visualises event data from json file.")
    parser.add_argument(
        "-d", "--datafile",
        type=str,
        default="Data/data.json",
        help="Path to the input ROOT data file."
    )
    parser.add_argument(
        "-s", "--skipevents",
        type=int,
        default=0,
        help="The number of events to skip visualising."
    )
    parser.add_argument(
        "-v", "--viewevents",
        type=int,
        default=2,
        help="The number of events to visualise."
    )
    parser.add_argument(
        "-n", "--markneutrino",
        action="store_true",
        default=False,
        help="Whether to mark particles that result from a neutrino interaction."
    )
    return parser.parse_args()

# Creates the hit masks for a given event
def createMasks(hitIds, particleTypes, systemTypes):
    neutrinoCode, PDGCodes, counters = np.array([str(id)[0] for id in hitIds]).astype(np.int32), np.array([str(id)[1:11] for id in hitIds]).astype(np.int32), np.array([str(id)[-5:] for id in hitIds]).astype(np.int32)
    systemTypes[0]["mask"] = np.full(len(hitIds), False)
    for pType in particleTypes:
        currentMask = (PDGCodes==pType["PDG"])
        pType["mask"] = currentMask
        systemTypes[0]["mask"] = systemTypes[0]["mask"] | currentMask
    systemTypes[1]["mask"] = ~systemTypes[0]["mask"]
    systemTypes[2]["mask"] = (neutrinoCode==8)
    return particleTypes, systemTypes

# Plots the colourised hits for a given event
def hitsPlot(hitPositions, particleTypes, systemTypes, markNeutrino, markerSize=0.2):
    if markNeutrino:
        neutrinoMask = systemTypes[2]["mask"]
        for pType in particleTypes:
            xCoords, zCoords = hitPositions[:,0][pType["mask"] & neutrinoMask], hitPositions[:,1][pType["mask"] & neutrinoMask]
            plt.scatter(xCoords, zCoords, s=4*markerSize, c="k", marker="x")
            xCoords, zCoords = hitPositions[:,0][pType["mask"] & ~neutrinoMask], hitPositions[:,1][pType["mask"] & ~neutrinoMask]
            plt.scatter(xCoords, zCoords, s=markerSize, c=pType["colour"], marker=".")
    else:
        for pType in particleTypes:
            xCoords, zCoords = hitPositions[:,0][pType["mask"]], hitPositions[:,1][pType["mask"]]
            plt.scatter(xCoords, zCoords, s=markerSize, c=pType["colour"], marker=".")
    miscType = systemTypes[1]
    xCoords, zCoords = hitPositions[:,0][miscType["mask"]], hitPositions[:,1][miscType["mask"]]
    plt.scatter(xCoords, zCoords, s=markerSize, c=miscType["colour"])

    plt.title("W View")
    plt.xlabel("X Position /mm")
    plt.ylabel("Z Position /mm")
    plt.show()
    return None

# Defines the particle types and then visualises each event
def main():
    args = parser()
    dataFile, skipEvents, viewEvents, markNeutrino = args.datafile, args.skipevents, args.viewevents, args.markneutrino

    particleTypes = [
        {"name": "proton", "PDG": 2212, "colour": "red", "mask": []},
        {"name": "chargedpion", "PDG": 211, "colour": "green", "mask": []},
        {"name": "muon", "PDG": 13, "colour": "blue", "mask": []}
    ]
    systemTypes = [
        {"name": "all", "PDG": -1, "colour": "grey", "mask": []},
        {"name": "misc", "PDG": -1, "colour": "grey", "mask": []},
        {"name": "parentneutrino", "PDG": -1, "colour": "grey", "mask": []}
    ]

    with open(dataFile, "r") as f:
        data = json.load(f)

    for i in range(skipEvents, min(len(data), skipEvents+viewEvents)):
        eventData = data[i]
        ids, energies, hitPositions, hitIds = eventData["ids"], eventData["energies"], np.array(eventData["hitPositions"]), eventData["hitIds"]
        particleTypesMasked, systemTypesMasked = createMasks(hitIds, particleTypes, systemTypes)
        hitsPlot(hitPositions, particleTypesMasked, systemTypesMasked, markNeutrino)

    return None
    
main()