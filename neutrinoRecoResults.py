import numpy as np
import matplotlib.pyplot as plt
import uproot
import json

def dataUnpack(filename="Data/data.root", treename="neutrino"):
    with uproot.open(f"{filename}:{treename}") as tree:
        purities = tree["purities"].array(library="np")
        completenesses = tree["completenesses"].array(library="np")
        eventIds = tree["eventIds"].array(library="np")
        numSlices = tree["numSlices"].array(library="np")
        neutrinoScores = tree["neutrinoScores"].array(library="np")
        numHits = tree["numHits"].array(library="np")
    
    print(f"There are {len(purities)} total events.")

    return np.array((purities, completenesses, eventIds, numSlices, neutrinoScores, numHits))

def plot(purities, completenesses, bins=20):
    reducedPurities, reducedCompletenesses = purities[purities>=0], completenesses[purities>=0]
    print(f"There are {len(reducedPurities)} selected events.")

    plt.figure(figsize=(10,6))
    plt.hist(reducedPurities, bins=bins, label="Purity", histtype="step", lw=2)
    plt.hist(reducedCompletenesses, bins=bins, label="Completeness", histtype="step", lw=2)

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.show()

def dataOutput(purities, completenesses, eventIds, numSlices, neutrinoScores, numHits, outputFile="Data/temp2.json"):
    events = []
    eventFile, priorId = 1, 0
    for i in range(len(eventIds)):
        if eventIds[i] < priorId:
            eventFile += 1
        event = {
            "eventId": i+1,
            "eventFile": eventFile,
            "eventNum": int(eventIds[i]),
            "purity": float(purities[i]),
            "completeness": float(completenesses[i]),
            "nuScore": float(neutrinoScores[i]),
            "numNuSlices": int(numSlices[i]),
            "numTrueHits": int(numHits[i])
        }
        events.append(event)
        priorId = eventIds[i]

    selectedEvents = [e for e in events if ((e["purity"]<0.7 or e["completeness"]<0.7) and e["nuScore"]>0.5 and e["numTrueHits"]>9)]
    print(f"Number of events: {len(events)}\nNumber of event failures: {len(selectedEvents)}")


    with open(outputFile, "w") as f:
        json.dump(selectedEvents, f, indent=4)
    with open("Data/temp1.json", "w") as f:
        json.dump(events, f, indent=4)
    print("Output file created")
    return None

def main():
    data = dataUnpack()
    #plot(*data)
    dataOutput(*data)
    return None

main()