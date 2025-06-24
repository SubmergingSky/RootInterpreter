import numpy as np
import uproot
import matplotlib.pyplot as plt

# Unpacks a root file and returns a list of event data
def inputUnpack(filename, treename="mc_info"):
    with uproot.open(f"{filename}:{treename}") as tree:
        ids = tree["ids"].array(library="np")
        energies = tree["energies"].array(library="np") #[GeV]
        hitPositions = np.transpose([tree["hitsX"].array(library="np"), tree["hitsY"].array(library="np"), tree["hitsZ"].array(library="np")]) #[mm]
        hitIds = tree["hitIds"].array(library="np") # 8/9 nnnnnnnnnn "PDG code" nnnnn "particle counter"

    return ids, energies, hitPositions, hitIds

# Creates the hit masks for a given event
def createMasks(hitIds, particleTypes, systemTypes):
    neutrinoCode, PDGCodes, counters = np.array([str(id)[0] for id in hitIds]).astype(np.int32), np.array([str(id)[1:11] for id in hitIds]).astype(np.int32), np.array([str(id)[-5] for id in hitIds]).astype(np.int32)
    systemTypes[0]["mask"] = np.full(len(hitIds), False)
    for pType in particleTypes:
        currentMask = (PDGCodes==pType["PDG"])
        pType["mask"] = currentMask
        systemTypes[0]["mask"] = systemTypes[0]["mask"] | currentMask
    systemTypes[1]["mask"] = ~systemTypes[0]["mask"]
    systemTypes[2]["mask"] = (neutrinoCode==8)
    return particleTypes, systemTypes

# Plots the colourised hits for a given event
def hitsPlot(hitPositions, particleTypes, systemTypes, markerSize=0.2, markNeutrino=True):
    if markNeutrino:
        neutrinoMask = systemTypes[2]["mask"]
        for pType in particleTypes:
            xCoords, zCoords = hitPositions[0][pType["mask"] & neutrinoMask], hitPositions[2][pType["mask"] & neutrinoMask]
            plt.scatter(xCoords, zCoords, s=4*markerSize, c="k", marker="x")
            xCoords, zCoords = hitPositions[0][pType["mask"] & ~neutrinoMask], hitPositions[2][pType["mask"] & ~neutrinoMask]
            plt.scatter(xCoords, zCoords, s=markerSize, c=pType["colour"], marker=".")
    else:
        for pType in particleTypes:
            xCoords, zCoords = hitPositions[0][pType["mask"]], hitPositions[2][pType["mask"]]
            plt.scatter(xCoords, zCoords, s=markerSize, c=pType["colour"], marker=".")
    miscType = systemTypes[1]
    xCoords, zCoords = hitPositions[0][miscType["mask"]], hitPositions[2][miscType["mask"]]
    plt.scatter(xCoords, zCoords, s=markerSize, c=miscType["colour"])

    plt.title("W View")
    plt.xlabel("X Position /mm")
    plt.ylabel("Z Position /mm")
    plt.show()
    return None

# Defines the particle types and then visualises each event
def main(dataFile="data.root", skipEvents=1, viewEvents=1):
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

    ids, energies, hitPositions, hitIds = inputUnpack(dataFile)
    for i in range(skipEvents, min(hitIds.shape[0], skipEvents+viewEvents)):
        particleTypesMasked, systemTypesMasked = createMasks(hitIds[i], particleTypes, systemTypes)
        hitsPlot(hitPositions[i], particleTypesMasked, systemTypesMasked)
    
    return None

main()