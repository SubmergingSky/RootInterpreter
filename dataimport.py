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

# Unpacks a root file and returns a list of event data
def dataUnpack(filename, treename):
    with uproot.open(f"{filename}:{treename}") as tree:
        ids = tree["ids"].array(library="np")
        energies = tree["energies"].array(library="np") #[GeV]
        hitIds = tree["hitIds"].array(library="np") # 8/9 nnnnnnnnnn "PDG code" nnnnn "particle counter"
        hitsX, hitsY, hitsZ = tree["hitsX"].array(library="np"), tree["hitsY"].array(library="np"), tree["hitsZ"].array(library="np") #[mm]

    return (ids, energies, hitIds, hitsX, hitsY, hitsZ)

# Outputs the unpacked data to a json file
def dataOutput(data, outputFile):
    ids, energies, hitIds, hitsX, hitsY, hitsZ = data
    allData = []
    for i in range(ids.shape[0]):
        hitPositions = np.column_stack((hitsX[i], hitsY[i], hitsZ[i]))
        eventData = {
            "ids": ids[i].tolist(),
            "energies": energies[i].tolist(),
            "hitIds": hitIds[i].tolist(),
            "hitPositions": hitPositions.tolist()
        }
        allData.append(eventData)
    
    with open(outputFile, "w") as f:
        json.dump(allData, f, indent=4)
    print("Output file created")

def main():
    args = parser()
    dataFile, treename, outputFile = args.datafile, args.treename, args.outputfile
    data = dataUnpack(dataFile, treename)
    dataOutput(data, outputFile)

main()