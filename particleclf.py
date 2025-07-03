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

def dataUnpack(dataFile):
    with open(f"Data/{dataFile}", "r") as f:
        data = json.load(f)
    
    clusters = np.array([cluster for cluster in data if cluster["PDGCode"] in [211, 13]])
    features = []
    for cluster in clusters:
        clusterFeatures = [cluster["numHits"], cluster["endpointsDistance"], cluster["linearRmsError"], cluster["meanEnergyDeposition"], cluster["rmsRateEnergyDeposition"], cluster["rmsHitGap"], cluster["transverseWidth"]]
        features.append(clusterFeatures)

    print(f"Data unpacked from {dataFile}")
    return np.array(features), clusters

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

def makeCosmicClf(xTrain, cTrain):
    yTrain = np.array([c["isFromNeutrino"] for c in cTrain]).astype(int) # (1) for beam particles, (0) for cosmics
    clf = makeModel(xTrain, yTrain)
    print("Cosmic CLF fitted\n")
    return clf

def makeBeamClf(xTrain, cTrain):
    beamMask = np.array([c["isFromNeutrino"] for c in cTrain])
    xTrain, cTrain = xTrain[beamMask], [c for (i, c) in enumerate(cTrain) if beamMask[i]]
    yTrain = np.array([c["PDGCode"]==211 for c in cTrain]).astype(int) # (1) for pions, (0) for muons
    clf = makeModel(xTrain, yTrain)
    print("Beam CLF fitted\n")
    return clf

def partialDependence(clf, xTrain, testingData):
    fig, ax = plt.subplots(figsize=(12, 6))
    PartialDependenceDisplay.from_estimator(clf, xTrain, features=[0,1,2], ax=ax)
    

    xTest, yTest = testingData
    permImportance = permutation_importance(clf, xTest, yTest, random_state=1, n_jobs=8, scoring="f1_macro")
    for i in permImportance.importances_mean.argsort()[::-1]:
        print(f"Feature {i:<2}: {permImportance.importances_mean[i]:.4f} +/- {permImportance.importances_std[i]:.4f}")

    plt.show()
    return None


def evaluation(cosmicClf, beamClf, testingData):
    def permImportance(clf, xTest, yTest):
        permImportance = permutation_importance(clf, xTest, yTest, random_state=1, n_jobs=8, scoring="f1_macro")
        for i in permImportance.importances_mean.argsort()[::-1]:
            print(f"Feature {i:<2}: {permImportance.importances_mean[i]:.4f} +/- {permImportance.importances_std[i]:.4f}")
        return None


    xTest, cTest = testingData
    cosmicPred, yTest = cosmicClf.predict(xTest), np.array([c["isFromNeutrino"] for c in cTest]).astype(int)
    f1score = f1_score(yTest, cosmicPred, average="macro")

    print("--- Classification Report (Stage 1) ---")
    print(classification_report(yTest, cosmicPred))
    print("--- Macro-Averaged F1 Score ---")
    print(f"Macro-Averaged F1-Score: {f1score:.3f}")
    permImportance(cosmicClf, xTest, yTest)

    beamMask = np.array([c["isFromNeutrino"] for c in cTest])
    xTestBeam, cTestBeam = xTest[beamMask], [c for (i, c) in enumerate(cTest) if beamMask[i]]
    beamPred, yTest = beamClf.predict(xTestBeam), np.array([c["PDGCode"]==211 for c in cTestBeam]).astype(int)
    f1score = f1_score(yTest, beamPred, average="macro")

    print("--- Classification Report (Stage 2) ---")
    print(classification_report(yTest, beamPred))
    print("--- Macro-Averaged F1 Score ---")
    print(f"Macro-Averaged F1-Score: {f1score:.3f}")
    permImportance(beamClf, xTestBeam, yTest)

    yTrue =  np.array([0 if not c["isFromNeutrino"] else c["PDGCode"] for c in cTest])
    combinedPred, beamMaskPred = np.zeros_like(cosmicPred), (cosmicPred==1)
    xTest = xTest[beamMaskPred]
    beamPred = beamClf.predict(xTest)
    combinedPred[beamMaskPred] = np.where(beamPred == 1, 211, 13)
    f1score = f1_score(yTrue, combinedPred, average="macro")

    print("--- Classification Report (Combined) ---")
    print(classification_report(yTrue, combinedPred))
    print("--- Macro-Averaged F1 Score ---")
    print(f"Macro-Averaged F1-Score: {f1score:.3f}")

    return None


def main():
    args = parser()
    trainingFile, testingFile, displayPartial = args.trainingfile, args.testingfile, args.displaypartial

    trainingData = dataUnpack(trainingFile)
    cosmicClf = makeCosmicClf(*trainingData) 
    beamClf = makeBeamClf(*trainingData)

    testingData = dataUnpack(testingFile)
    evaluation(cosmicClf, beamClf, testingData)
    
    

    
    #if displayPartial: partialDependence(clf, trainingData[0], testingData)

    return None

main()