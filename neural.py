import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter


def parser():
    parser = argparse.ArgumentParser(description="Uses ANNs to make predictions on particle classifications.")
    parser.add_argument(
        "-d", "--trainingfile",
        type=str,
        default="featured_data.json",
        help="Path to the input training file."
    )
    parser.add_argument(
        "-t", "--testingfile",
        type=str,
        default="testing_data.json",
        help="Path to the input testing file."
    )
    return parser.parse_args()

def dataUnpack(dataFile):
    with open(f"Data/{dataFile}", "r") as f:
        data = json.load(f)
    
    clusters = [cluster for cluster in data if cluster["PDGCode"] in [211, 13]]
    targets = np.array([cluster["PDGCode"]==211 for cluster in clusters]).astype(int)
    features = []
    for cluster in clusters:
        clusterFeatures = [cluster["linearRmsError"], cluster["transverseWidth"], cluster["meanEnergyDeposition"], cluster["rmsRateEnergyDeposition"], cluster["endpointsDistance"], cluster["numHits"], cluster["rmsHitGap"]]
        features.append(clusterFeatures)

    return (features, targets)

def classification(data, resample=True):
    features, targets = data
    print(f"Original dataset shape: {Counter(targets)}")
    if resample:
        smote = BorderlineSMOTE(random_state=1)
        features, targets = smote.fit_resample(features, targets)
        print(f"Resampled dataset shape: {Counter(targets)}")

    clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(50,10), max_iter=1500, random_state=1)
    clf.fit(features, targets)
    return clf

def evaluation(clf, testingData, displayPartial=False):
    prediction = clf.predict(testingData[0])
    f1score = f1_score(testingData[1], prediction, average="macro")

    print("--- Classification Report ---")
    print(classification_report(testingData[1], prediction))
    print("--- Macro-Averaged F1 Score ---")
    print(f"Macro-Averaged F1-Score: {f1score:.3f}")

    if displayPartial:
        fig, ax = plt.subplots(figsize=(12, 6))
        PartialDependenceDisplay.from_estimator(clf, testingData[0], features=[0,1,2,3,4,5,6], ax=ax)
        plt.show()

    return None


def main():
    args = parser()
    dataFile, testingFile = args.trainingfile, args.testingfile

    trainingData = dataUnpack(dataFile)
    print("Data unpacked")

    clf = classification(trainingData)
    print("Neural network fitted")

    testingData = dataUnpack(testingFile)
    evaluation(clf, testingData)

    return None
    
main()