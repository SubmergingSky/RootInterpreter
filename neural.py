import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline

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
    parser.add_argument(
        "-p", "--displaypartial",
        action="store_true",
        default=False,
        help="Whether to show the partial dependance plots."
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
        clusterFeatures = [cluster["transverseWidth"], cluster["meanEnergyDeposition"], cluster["endpointsDistance"]]
        features.append(clusterFeatures)

    print("Data unpacked")
    return features, targets

def makeModel(xTrain, yTrain):
    print(f"Original dataset shape: {Counter(yTrain)}")
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("sampler", BorderlineSMOTE(random_state=1)),
        ("mlp", MLPClassifier(max_iter=1500, random_state=1, early_stopping=True))
        ])
    param_grid = {
        "mlp__hidden_layer_sizes": [(50,), (100,), (90,45), (50,10), (100,50), (70,40)]
    }

    gridSearch = GridSearchCV(pipeline, param_grid, scoring="f1_macro", n_jobs=8)
    print("Beginning grid search")
    gridSearch.fit(xTrain, yTrain)
    print(f"Grid search complete. Best layer setup identified: {gridSearch.best_params_}")
    clf = gridSearch.best_estimator_

    return clf

def makeModelSpecific(xTrain, yTrain):
    print(f"Original dataset shape: {Counter(yTrain)}")
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("sampler", BorderlineSMOTE(random_state=1)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(90,45), max_iter=1500, random_state=1))
        ])

    clf = pipeline.fit(xTrain, yTrain)
    print("Fitting complete")

    return clf

# Evaluates the success of the model.
def evaluation(clf, data):
    xTest, yTest = data
    prediction = clf.predict(xTest)
    f1score = f1_score(yTest, prediction, average="macro")

    print("--- Classification Report ---")
    print(classification_report(yTest, prediction))
    print("--- Macro-Averaged F1 Score ---")
    print(f"Macro-Averaged F1-Score: {f1score:.3f}")

    return None

def partialDependence(clf, xTrain, testingData):
    fig, ax = plt.subplots(figsize=(12, 6))
    PartialDependenceDisplay.from_estimator(clf, xTrain, features=[0,1,2], ax=ax)
    

    xTest, yTest = testingData
    permImportance = permutation_importance(clf, xTest, yTest, random_state=1, n_jobs=8, scoring="f1_macro")
    for i in permImportance.importances_mean.argsort()[::-1]:
        print(f"Feature {i:<2}: {permImportance.importances_mean[i]:.4f} +/- {permImportance.importances_std[i]:.4f}")

    plt.show()
    return None


def main():
    args = parser()
    trainingFile, testingFile, displayPartial = args.trainingfile, args.testingfile, args.displaypartial

    trainingData = dataUnpack(trainingFile)
    clf = makeModel(*trainingData)
    #clf = makeModelSpecific(*trainingData)

    testingData = dataUnpack(testingFile)
    evaluation(clf, testingData)
    if displayPartial: partialDependence(clf, trainingData[0], testingData)

    return None
    
main()