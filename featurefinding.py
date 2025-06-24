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
    return parser.parse_args()

# Opens the data file and returns a list of particle counters and a list of hit positions.
def getData(dataFile):
    with open(dataFile, "r") as f:
        data = json.load(f)
    
    counters, hits = [], []
    for event in data:
        hitIds, hitPositions = event["hitIds"], np.array(event["hitPositions"])
        for id in hitIds: counters.append(int(str(id)[-5:]))
        for hit in hitPositions: hits.append(hit)

    return (counters, hits)

def main():
    args = parser()
    dataFile = args.datafile

    data = getData(dataFile)


main()