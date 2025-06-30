import numpy as np
from sklearn.neural_network import MLPClassifier
import json
import argparse

def parser():
    parser = argparse.ArgumentParser(description="Uses ANNs to make predictions on particle classifications.")
    parser.add_argument(
        "-d", "--trainingfile",
        type=str,
        default="Data/featured_data.json",
        help="Path to the input training file."
    )
    parser.add_argument(
        "-t", "--testingfile",
        type=str,
        default="Data/testing_data.json",
        help="Path to the input testing file."
    )
    return parser.parse_args()

def dataUnpack(dataFile):
    with open(dataFile, "r") as f:
        data = json.load(f)
    
    clusters = [cluster for cluster in data if cluster["PDGCode"] in [211, 13]]
    targets = np.array([cluster["PDGCode"]==211 for cluster in clusters]).astype(int)
    features = []
    for cluster in clusters:
        clusterFeatures = [cluster["linearRmsError"], cluster["meanEnergyDeposition"], cluster["rmsRateEnergyDeposition"], cluster["endpointsDistance"], cluster["numHits"], cluster["rmsHitGap"]]
        features.append(clusterFeatures)

    return (features, targets)

def classification(data):
    features, targets = data
    clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(3), max_iter=500, random_state=1)
    clf.fit(features, targets)
    return clf

def main():
    args = parser()
    dataFile, testingFile = args.trainingfile, args.testingfile

    trainingData = dataUnpack(dataFile)
    print("Data unpacked")

    clf = classification(trainingData)
    print("Neural network fitted")

    testingData = dataUnpack(testingFile)
    print(clf.predict(testingData[0]))
    print("Accuracy:  ", clf.score(*testingData))


    

main()