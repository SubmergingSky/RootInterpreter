import numpy as np
import matplotlib.pyplot as plt, matplotlib.patches as patch, matplotlib.collections as collections
import json
import argparse
import copy

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
        default=2,
        help="The number of events to visualise."
    )
    parser.add_argument(
        "-n", "--markneutrino",
        action="store_true",
        default=False,
        help="Whether to mark particles that result from a neutrino interaction."
    )
    parser.add_argument(
        "-c", "--markcosmics",
        action="store_true",
        default=False,
        help="Whether to mark particles that have been deemed clear cosmics."
    )
    parser.add_argument(
        "-3", "--threedplot",
        action="store_true",
        default=False,
        help="Whether to use a 3D plot."
    )
    return parser.parse_args()

# Creates the hit masks for a given event
def createMasks(neutrinoCodes, clearCosmicCodes, PDGCodes, particleTypes, systemTypes):
    systemTypes[0]["mask"] = np.full(len(PDGCodes), False)
    for pType in particleTypes:
        currentMask = (PDGCodes==pType["PDG"])
        pType["mask"] = currentMask
        systemTypes[0]["mask"] = systemTypes[0]["mask"] | currentMask
    systemTypes[1]["mask"], systemTypes[2]["mask"], systemTypes[3]["mask"] = ~systemTypes[0]["mask"], (neutrinoCodes), (clearCosmicCodes)

    return particleTypes, systemTypes

# Plots a given detector event.
def eventPlot(hitPositions, hitGeometries, particleTypes, systemTypes, markNeutrino, markCosmics, threeD):
    def plotHits(hitPositions, hitGeometries, mask, colour):
        centresX, centresY, centresZ = hitPositions[:,0][mask], hitPositions[:,1][mask], hitPositions[:,2][mask]
        if threeD:
            ax.scatter(centresX, centresY, centresZ, c=colour, s=0.4)
        else:
            cellSizeX, cellSizeZ = hitGeometries[:,0][mask], hitGeometries[:,2][mask]
            patches = []
            for i in range(len(centresX)):
                cornerX, cornerZ = centresX[i] - cellSizeX[i]/2, centresZ[i] - cellSizeZ[i]/2
                rect = patch.Rectangle((cornerX, cornerZ), cellSizeX[i], cellSizeZ[i])
                patches.append(rect)
            ax.add_collection(collections.PatchCollection(patches, facecolor=colour, edgecolor=colour, linewidth=0.5, alpha=0.7))
        return None
    
    if threeD:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X Position /mm")
        ax.set_ylabel("Y Position /mm")
        ax.set_zlabel("Z Position /mm")
        ax.set_title('3D Event Plot')
    else:
        fig, ax = plt.subplots(figsize=(10,8))
        ax.set_title("W View")
        ax.set_xlabel("X Position /mm")
        ax.set_ylabel("Z Position /mm")
    
    if markNeutrino:
        neutrinoMask = systemTypes[2]["mask"]
        for pType in particleTypes:
            plotHits(hitPositions, hitGeometries, (pType["mask"] & neutrinoMask), "black")
            plotHits(hitPositions, hitGeometries, (pType["mask"] & ~neutrinoMask), pType["colour"])
    elif markCosmics:
        cosmicMask = systemTypes[3]["mask"]
        for pType in particleTypes:
            plotHits(hitPositions, hitGeometries, (pType["mask"] & cosmicMask), "black")
            plotHits(hitPositions, hitGeometries, (pType["mask"] & ~cosmicMask), pType["colour"])
    else:
        for pType in particleTypes:
            plotHits(hitPositions, hitGeometries, pType["mask"], pType["colour"])
    miscType = systemTypes[1]
    plotHits(hitPositions, hitGeometries, miscType["mask"], miscType["colour"])

    ax.autoscale_view()
    plt.show()
    return None

# Plots the hitmap of a given cluster.
def particlePlot(cluster, i, threeD=False):
    if threeD:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X Position /mm")
        ax.set_ylabel("Y Position /mm")
        ax.set_zlabel("Z Position /mm")
        ax.set_title('3D Event Plot')
    else:
        fig, ax = plt.subplots(figsize=(10,8))
        ax.set_title(f"Plot {i+1}   {cluster["eventId"]}")
        ax.set_xlabel("X Position /mm")
        ax.set_ylabel("Z Position /mm")


    hitPositions, hitGeometries, MCHits = np.array(cluster["hits"]), np.array(cluster["hitGeometries"]), np.array(cluster["MCHits"])
    centresX, centresY, centresZ = hitPositions[:,0], hitPositions[:,1], hitPositions[:,2]
    MCX, MCY, MCZ = MCHits[:,0], MCHits[:,1], MCHits[:,2]
    if threeD:
        ax.scatter(centresX, centresY, centresZ, s=0.4)
    else:
        cellSizeX, cellSizeZ = hitGeometries[:,0], hitGeometries[:,2]
        patches = []
        for i in range(len(centresX)):
            cornerX, cornerZ = centresX[i] - cellSizeX[i]/2, centresZ[i] - cellSizeZ[i]/2
            rect = patch.Rectangle((cornerX, cornerZ), cellSizeX[i], cellSizeZ[i])
            patches.append(rect)
        ax.add_collection(collections.PatchCollection(patches, facecolor="b", edgecolor="b", linewidth=0.5, alpha=0.7))
        ax.scatter(MCX, MCZ, s=1, c="green")


    #ax.set_xlim((-234,234))
    #ax.set_ylim((5, 505))
    ax.autoscale_view()
    plt.show()
    return None


# TEST FUNCTION
def particleTest():
    with open("Data/full_featured_data1_10.json", "r") as f:
        data = json.load(f)

    validClusters = []
    for cluster in data:
        if cluster:
            validClusters.append(cluster)
        else:
            continue
    reducedValidClusters = copy.deepcopy(validClusters)
    for c in reducedValidClusters:
        del c["hits"], c["hitGeometries"], c["inputEnergies"], c["MCHits"]
    
    with open("Data/temp.json", "w") as f:
        json.dump(reducedValidClusters[0:5], f, indent=4)
    for i in range(5):
        particlePlot(validClusters[i], i)


    return None

# Defines the particle types and then visualises each event
def main():
    args = parser()
    dataFile, skipEvents, viewEvents, markNeutrino, markCosmics, threeDPlot = args.datafile, args.skipevents, args.viewevents, args.markneutrino, args.markcosmics, args.threedplot

    particleTypes = [
        {"name": "proton", "PDG": 2212, "colour": "pink", "mask": []},
        {"name": "chargedpion", "PDG": 211, "colour": "green", "mask": []},
        {"name": "muon", "PDG": 13, "colour": "purple", "mask": []},
        {"name": "photon", "PDG": 22, "colour": "orange", "mask": []},
        {"name": "electron", "PDG": 11, "colour": "blue", "mask": []},
        {"name": "muonneutrino", "PDG": 14, "colour": "yellow", "mask": []}
    ]
    systemTypes = [
        {"name": "all", "PDG": -1, "colour": "grey", "mask": []},
        {"name": "misc", "PDG": -1, "colour": "grey", "mask": []},
        {"name": "parentneutrino", "PDG": -1, "colour": "grey", "mask": []},
        {"name": "clearcosmic", "PDG": -1, "colour": "grey", "mask": []}
    ]

    with open(dataFile, "r") as f:
        data = json.load(f)

    eventIds = np.unique([cluster.get("eventId") for cluster in data])
    for i in range(skipEvents, min(len(eventIds), skipEvents+viewEvents)):
        eventClusters = [cluster for cluster in data if cluster.get("eventId")==eventIds[i]]
        isFromNeutrinos, isClearCosmic, PDGCodes, hitPositions, hitGeometries = [], [], [], [], []
        for cluster in eventClusters:
            for hit, hitGeometry in zip(cluster["hits"], cluster["hitGeometries"]):
                isFromNeutrinos.append(cluster["isFromNeutrino"])
                isClearCosmic.append(cluster["isClearCosmic"])
                PDGCodes.append(cluster["PDGCode"])
                hitPositions.append(hit)
                hitGeometries.append(hitGeometry)
        isFromNeutrinos, isClearCosmic, PDGCodes, hitPositions, hitGeometries = np.array(isFromNeutrinos), np.array(isClearCosmic), np.array(PDGCodes), np.array(hitPositions), np.array(hitGeometries)
        particleTypesMasked, systemTypesMasked = createMasks(isFromNeutrinos, isClearCosmic, PDGCodes, particleTypes, systemTypes)
        eventPlot(hitPositions, hitGeometries, particleTypesMasked, systemTypesMasked, markNeutrino, markCosmics, threeDPlot)

    return None
    
#main()
particleTest()