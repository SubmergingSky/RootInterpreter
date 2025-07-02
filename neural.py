import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV

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

# Unpacks a json file, selects only pions and muons, and returns the needed data.
# Target (1) is set as pion.
def dataUnpack(dataFile):
    with open(f"Data/{dataFile}", "r") as f:
        data = json.load(f)
    
    clusters = [cluster for cluster in data if cluster["PDGCode"] in [211, 13]]
    targets = np.array([cluster["PDGCode"]==211 for cluster in clusters]).astype(int)
    features = []
    for cluster in clusters:
        clusterFeatures = [cluster["linearRmsError"], cluster["transverseWidth"], cluster["meanEnergyDeposition"], cluster["rmsRateEnergyDeposition"], cluster["endpointsDistance"], cluster["rmsHitGap"]]
        features.append(clusterFeatures)

    print("Data unpacked")
    return (features, targets)

# Resamples and scales the initial data.
def preprocessing(data):
    features, targets = data
    print(f"Original dataset shape: {Counter(targets)}")

    smote = BorderlineSMOTE(random_state=1)
    features, targets = smote.fit_resample(features, targets)
    print(f"Resampled dataset shape: {Counter(targets)}")

    scaler = RobustScaler().fit(features)
    features = scaler.transform(features)

    return (features, targets), scaler

# Produces a fitted NN for given data.
def classification(features, targets):
    clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(50,10), max_iter=1500, random_state=1, early_stopping=True).fit(features, targets)

    print("Neural network fitted")
    return clf

# Evaluates the success of the model.
def evaluation(clf, scaler, data, displayPartial=False):
    features, targets = data
    prediction = clf.predict(scaler.transform(features))
    f1score = f1_score(targets, prediction, average="macro")

    print("--- Classification Report ---")
    print(classification_report(targets, prediction))
    print("--- Macro-Averaged F1 Score ---")
    print(f"Macro-Averaged F1-Score: {f1score:.3f}")

    if displayPartial:
        fig, ax = plt.subplots(figsize=(12, 6))
        PartialDependenceDisplay.from_estimator(clf, features, features=[0,1,2,3,4,5], ax=ax)
        plt.show()

    return None

def layerTesting(data):
    features, targets = data
    print(f"Original dataset shape: {Counter(targets)}")
    smote = BorderlineSMOTE(random_state=1)
    features, targets = smote.fit_resample(features, targets)
    print(f"Resampled dataset shape: {Counter(targets)}")
    scaler = RobustScaler().fit(features)
    features = scaler.transform(features)

    param_grid = {
        'hidden_layer_sizes': [
            (50,),
            (100,),
            (50, 25),
            (100, 50),
            (50, 10)
        ]
    }
    gridSearch = GridSearchCV(MLPClassifier(max_iter=1500, random_state=1), param_grid, cv=3, scoring="f1_macro")
    gridSearch.fit(features, targets)
    print(f"Best parameters found: {gridSearch.best_params_}")
    return None


def main():
    args = parser()
    trainingFile, testingFile = args.trainingfile, args.testingfile

    trainingData = dataUnpack(trainingFile)
    trainingData, scaler = preprocessing(trainingData)
    
    clf = classification(*trainingData)
    
    testingData = dataUnpack(testingFile)
    evaluation(clf, scaler, testingData)

    return None
    
main()