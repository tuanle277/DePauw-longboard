import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.tree import plot_confusion_matrix

def train_models(traindata, labels, config):
    X_train, y_train = traindata, labels
    classifiers = config['models']['classifiers']
    
    for classifier_name, classifier_params in classifiers.items():
        classifier = RandomForestClassifier(**classifier_params)
        X_train = SelectKBest(f_classif, k=config['models']['num_features']).fit_transform(X_train, y_train)
        start_time = time.time()
        classifier.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"Trained {classifier_name} in {end_time - start_time:.2f} seconds")
        # Save the model if needed

def batch_classify(data_subsets, num_features, verbose=True):
    """
    Trains and evaluates classifiers on the provided data subsets.

    Args:
        data_subsets (tuple): A tuple containing X_train, Y_train, X_test, Y_test.
        num_features (int): The number of features to select.
        verbose (bool, optional): If True, prints additional information. Defaults to True.

    Returns:
        dict: A dictionary containing the trained models, classification reports, and predictions.
    """
    X_train, Y_train, X_test, Y_test = data_subsets
    feature_names = ["AccelDiff", "xAmp", "xFreq", "zAmp", "zFreq", "rmsay", "rmsxx", "rmsxz"]
    labels = ["pumping", "pushing", "coasting"]

    dict_classifiers = {
        "Random Forest": RandomForestClassifier(penalty='l1', solver='liblinear', C=1.0),
    }

    dict_models = {}

    for classifier_name, classifier in dict_classifiers.items():
        start_time = time.time()
        X_train_selected = SelectKBest(f_classif, k=num_features).fit_transform(X_train, Y_train)
        X_test_selected = SelectKBest(f_classif, k=num_features).fit_transform(X_test, Y_test)
        classifier.fit(X_train_selected, Y_train)
        end_time = time.time()

        if classifier_name in {"Decision Tree", "Random Forest"}:
            importance = classifier.feature_importances_
            for i, v in enumerate(importance):
                print(f'Feature: {i}, Score: {v:.5f}')

            plt.rcParams.update({'font.size': 12})
            plt.rcParams['figure.figsize'] = [9, 7]
            plt.bar(feature_names, importance)
            plt.tight_layout()
            plt.title("Chart of Feature Importance")
            plt.show()

        train_time = end_time - start_time
        train_score = classifier.score(X_train_selected, Y_train)
        test_score = classifier.score(X_test_selected, Y_test)
        preds = classifier.predict(X_test_selected)

        unique_values, counts = np.unique(preds, return_counts=True)
        print("Predictions")
        for i in range(len(counts)):
            print(f"{unique_values[i]}: {counts[i] / sum(counts)}")

        unique_values, counts = np.unique(Y_test, return_counts=True)
        print("Actual")
        for i in range(len(counts)):
            print(f"{unique_values[i]}: {counts[i] / sum(counts)}")

        report = classification_report(Y_test, preds, output_dict=True)
        kappa = cohen_kappa_score(Y_test, preds)
        metr = confusion_matrix(Y_test, preds)
        plot_confusion_matrix(classifier, X_test_selected, Y_test, normalize='true', display_labels=labels, cmap=plt.cm.Blues)
        plt.title(f"{classifier_name} Confusion Matrix")
        plt.show()

        dict_models[classifier_name] = {
            'model': classifier,
            'train_score': train_score,
            'test_score': test_score,
            'train_time': train_time,
            'classification_report': report,
            'predictions': preds
        }

        if verbose:
            print(f"Trained {classifier_name} in {train_time:.2f} seconds.")

    return dict_models