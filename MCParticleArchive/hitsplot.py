import numpy as np
import matplotlib.pyplot as plt, matplotlib.patches as patch, matplotlib.collections as collections
import json
import argparse

def parser():
    parser = argparse.ArgumentParser(description="Visualises event data from json file.")
    parser.add_argument(
        "-d", "--datafile",
        type=str,
        default="Data/data.json",
        help="Path to the input data file."
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
        default=1,
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
def createMasks(neutrinoCodes, PDGCodes, particleTypes, systemTypes):
    systemTypes[0]["mask"] = np.full(len(PDGCodes), False)
    for pType in particleTypes:
        currentMask = (PDGCodes==pType["PDG"])
        pType["mask"] = currentMask
        systemTypes[0]["mask"] = systemTypes[0]["mask"] | currentMask
    systemTypes[1]["mask"] = ~systemTypes[0]["mask"]
    systemTypes[2]["mask"] = (neutrinoCodes)

    return particleTypes, systemTypes

# Plots the colourised hits for a given event.
def eventPlot(hitPositions, hitGeometries, particleTypes, systemTypes, markNeutrino):
    def patchesCreation(hitPositions, hitGeometries, mask, colour):
        centresX, centresZ = hitPositions[:,0][mask], hitPositions[:,1][mask]
        widths, heights = hitGeometries[:,0][mask], hitGeometries[:,1][mask]
        patches = []
        for i in range(len(centresX)):
            cornerX, cornerZ = centresX[i] - widths[i]/2, centresZ[i] - heights[i]/2
            rect = patch.Rectangle((cornerX, cornerZ), widths[i], heights[i])
            patches.append(rect)
        ax.add_collection(collections.PatchCollection(patches, facecolor=colour, edgecolor=colour, linewidth=0.5, alpha=0.7))
        return None
    
    fig, ax = plt.subplots(figsize=(10,8))
    if markNeutrino:
        neutrinoMask = systemTypes[2]["mask"]
        for pType in particleTypes:
            patchesCreation(hitPositions, hitGeometries, (pType["mask"] & neutrinoMask), "black")
            patchesCreation(hitPositions, hitGeometries, (pType["mask"] & ~neutrinoMask), pType["colour"])
    else:
        for pType in particleTypes:
            patchesCreation(hitPositions, hitGeometries, pType["mask"], pType["colour"])
    miscType = systemTypes[1]
    patchesCreation(hitPositions, hitGeometries, miscType["mask"], miscType["colour"])

    ax.autoscale_view()
    ax.set_title("W View")
    ax.set_xlabel("X Position /mm")
    ax.set_ylabel("Z Position /mm")
    plt.show()

    return None 

# Plots the hitmap of a given cluster.
def particlePlot(cluster):
    xCoords, zCoords = np.array(cluster["hits"])[:,0], np.array(cluster["hits"])[:,1]
    plt.scatter(xCoords, zCoords, s=0.4)
    plt.title(f"{cluster["eventId"]}   {cluster["hitId"]}")
    plt.show()
    return None


# TEST FUNCTION
def particleTest():
    with open("Data/featured_data.json", "r") as f:
        data = json.load(f)

    validClusters = []
    for cluster in data:
        if cluster["PDGCode"]==13 and cluster["linearRmsError"]>5:
            validClusters.append(cluster)
        else:
            continue
    with open("Data/temp.json", "w") as f:
        json.dump(validClusters[0:5], f, indent=4)
    for i in range(5):
        particlePlot(validClusters[i])

    return None

# Defines the particle types and then visualises each event
def main():
    args = parser()
    dataFile, skipEvents, viewEvents, markNeutrino = args.datafile, args.skipevents, args.viewevents, args.markneutrino

    particleTypes = [
        {"name": "proton", "PDG": 2212, "colour": "red", "mask": []},
        {"name": "chargedpion", "PDG": 211, "colour": "green", "mask": []},
        {"name": "muon", "PDG": 13, "colour": "blue", "mask": []},
        {"name": "photon", "PDG": 22, "colour": "orange", "mask": []}
    ]
    systemTypes = [
        {"name": "all", "PDG": -1, "colour": "grey", "mask": []},
        {"name": "misc", "PDG": -1, "colour": "grey", "mask": []},
        {"name": "parentneutrino", "PDG": -1, "colour": "grey", "mask": []}
    ]

    with open(dataFile, "r") as f:
        data = json.load(f)

    eventIds = np.unique([cluster.get("eventId") for cluster in data])
    for i in range(skipEvents, min(len(eventIds), skipEvents+viewEvents)):
        eventClusters = [cluster for cluster in data if cluster.get("eventId")==eventIds[i]]
        isFromNeutrinos, PDGCodes, hitPositions, hitGeometries = [], [], [], []
        for cluster in eventClusters:
            for hit, hitGeometry in zip(cluster["hits"], cluster["hitGeometries"]):
                isFromNeutrinos.append(cluster["isFromNeutrino"])
                PDGCodes.append(cluster["PDGCode"])
                hitPositions.append(hit)
                hitGeometries.append(hitGeometry)
        isFromNeutrinos, PDGCodes, hitPositions, hitGeometries = np.array(isFromNeutrinos), np.array(PDGCodes), np.array(hitPositions), np.array(hitGeometries)

        particleTypesMasked, systemTypesMasked = createMasks(isFromNeutrinos, PDGCodes, particleTypes, systemTypes)
        eventPlot(hitPositions, hitGeometries, particleTypesMasked, systemTypesMasked, markNeutrino)

    return None
    
main()