import os

from utils import create_folder_if_not_exists
from sklearn.model_selection import train_test_split

from modules.data.data_processing import get_features_train_data
from modules.training.train import batch_classify
from modules.training.models import dict_classifiers
from modules.plots.plot_utils import display_dict_models

# def main():
#     config = load_config('config/config.yaml')
#     create_folder_if_not_exists(config['output_folder'])

#     # Gather data
#     traindata, labels = get_features_train_data(config['data']['window_length'])

#     # Train and evaluate models
#     train_models(traindata, labels, config)
#     evaluate_models(config)

def main():

    num_samples = 1

    final = {x: {'model': x, 'train_score': 0, 'test_score': 0, 'train_time': 0} for x in dict_classifiers}

    # Gather data in windows of 3s
    traindata, labels = get_features_train_data(3)

    num_features = traindata.shape[-1]

    macro_f1, micro_f1, avg_recall, avg_precision, accuracy, train_time = 0, 0, 0, 0, 0, 0
    x_train, x_test, y_train, y_test = train_test_split(traindata, labels, test_size=0.8, random_state=0)
    dataSubsets = (x_train, y_train, x_test, y_test)
    trainResults, report, preds = batch_classify(dataSubsets, num_features)

    for key in final:
        for k in final[key]:
            if type(final[key][k]) != str:
                final[key][k] += trainResults[key][k]

    macro_f1 += report["macro avg"]["f1-score"]
    micro_f1 += report["weighted avg"]["f1-score"]
    avg_recall += (report["macro avg"]["recall"] + report["weighted avg"]["recall"]) / 2
    avg_precision += (report["macro avg"]["precision"] + report["weighted avg"]["precision"]) / 2
    accuracy += report["accuracy"]

    print()
    print(macro_f1/num_samples, micro_f1/num_samples, avg_recall/num_samples, avg_precision/num_samples, accuracy/num_samples, train_time)
    display_dict_models({x: {i: final[x][i] / num_samples for i in final[x] if type(final[x][i]) != str} for x in final})

if __name__ == "__main__":
    main()