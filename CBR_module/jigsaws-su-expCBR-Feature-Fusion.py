import numpy as np
import random
import time
from itertools import chain
import os
import pandas as pd
import collections
import re
import math
import matplotlib
import matplotlib.pyplot as plt
import json
import shutil
from sktime.datatypes import convert
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sktime.classification.deep_learning.fcn import FCNClassifier
from tensorflow import keras
from sktime.classification.deep_learning.resnet import ResNetClassifier
from tensorflow.keras.models import Model
import argparse
import datetime
import pytz
import sys
sys.path.append('./')
###
from datasets.jigsaws_su import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tsfresh.feature_extraction import ComprehensiveFCParameters
import numpy as np
from sktime.datatypes import convert_to
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.preprocessing import LabelEncoder
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
###
from CBR_module.utils import cosine_similarity, rank_by_similarity, evaluate_test_vectors_to_csv, evaluate_test_vectors_with_counterfactuals
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler

###
parser = argparse.ArgumentParser(description='JIGSAWS Suturing CBR Feature Fusion')
parser.add_argument('--validation_option', type=str, default='LOSO1', help='Cross Validation Options')
args = parser.parse_args()
###

# temporal view: resnet
resnet_su_train_feature_vectors = np.load('./IJCAI-24/src/results/jigsaws/' + 'resnet_su_' +  str(args.validation_option) + '_train_feature_vectors.npy', allow_pickle=True)
resnet_su_validation_feature_vectors = np.load('./IJCAI-24/src/results/jigsaws/' + 'resnet_su_' +  str(args.validation_option) + '_validation_feature_vectors.npy', allow_pickle=True)
resnet_su_test_feature_vectors = np.load('./IJCAI-24/src/results/jigsaws/' + 'resnet_su_' +  str(args.validation_option) + '_test_feature_vectors.npy', allow_pickle=True)

print("resnet_su_train_feature_vectors shape:  ", resnet_su_train_feature_vectors.shape)
print("resnet_su_validation_feature_vectors shape:  ", resnet_su_validation_feature_vectors.shape)
print("resnet_su_test_feature_vectors shape:  ", resnet_su_test_feature_vectors.shape)
print("resnet_su_test_feature_vectors top 10:  ", resnet_su_test_feature_vectors[:10])

# temporal view: ROCKET
rocket_su_train_feature_vectors = np.load('./IJCAI-24/src/results/jigsaws/' + 'rocket_su_' +  str(args.validation_option) + '_train_feature_vectors.npy', allow_pickle=True)
rocket_su_validation_feature_vectors = np.load('./IJCAI-24/src/results/jigsaws/' + 'rocket_su_' +  str(args.validation_option) + '_validation_feature_vectors.npy', allow_pickle=True)
rocket_su_test_feature_vectors = np.load('./IJCAI-24/src/results/jigsaws/' + 'rocket_su_' +  str(args.validation_option) + '_test_feature_vectors.npy', allow_pickle=True)

print("rocket_su_train_feature_vectors shape:  ", rocket_su_train_feature_vectors.shape)
print("rocket_su_validation_feature_vectors shape:  ", rocket_su_validation_feature_vectors.shape)
print("rocket_su_test_feature_vectors shape:  ", rocket_su_test_feature_vectors.shape)


# shapelet view
shapelet_su_train_feature_vectors = np.load('./IJCAI-24/src/results/jigsaws/' + 'shapelet_su_' +  str(args.validation_option) + '_train_feature_vectors.npy', allow_pickle=True)
shapelet_su_validation_feature_vectors = np.load('./IJCAI-24/src/results/jigsaws/' + 'shapelet_su_' +  str(args.validation_option) + '_validation_feature_vectors.npy', allow_pickle=True)
shapelet_su_test_feature_vectors = np.load('./IJCAI-24/src/results/jigsaws/' + 'shapelet_su_' +  str(args.validation_option) + '_test_feature_vectors.npy', allow_pickle=True)

print("shapelet_su_train_feature_vectors shape:  ", shapelet_su_train_feature_vectors.shape)
print("shapelet_su_validation_feature_vectors shape:  ", shapelet_su_validation_feature_vectors.shape)
print("shapelet_su_test_feature_vectors shape:  ", shapelet_su_test_feature_vectors.shape)
print("shapelet_su_test_feature_vectors top 10:  ", shapelet_su_test_feature_vectors[:10])

# frequency view
fft_su_train_feature_vectors = np.load('./IJCAI-24/src/results/jigsaws/' + 'fft_su_' +  str(args.validation_option) + '_train_feature_vectors.npy', allow_pickle=True)
fft_su_validation_feature_vectors = np.load('./IJCAI-24/src/results/jigsaws/' + 'fft_su_' +  str(args.validation_option) + '_validation_feature_vectors.npy', allow_pickle=True)
fft_su_test_feature_vectors = np.load('./IJCAI-24/src/results/jigsaws/' + 'fft_su_' +  str(args.validation_option) + '_test_feature_vectors.npy', allow_pickle=True)

print("fft_su_train_feature_vectors shape:  ", fft_su_train_feature_vectors.shape)
print("fft_su_validation_feature_vectors shape:  ", fft_su_validation_feature_vectors.shape)
print("fft_su_test_feature_vectors shape:  ", fft_su_test_feature_vectors.shape)
print("fft_su_test_feature_vectors top 10:  ", fft_su_test_feature_vectors[:10])

####
# Concatenate the feature vectors:
concatenated_feature_vectors_su_train = np.concatenate(
    (fft_su_train_feature_vectors,
     resnet_su_train_feature_vectors,
     rocket_su_train_feature_vectors,
     shapelet_su_train_feature_vectors),
    axis=1
)
# Check the shape of the concatenated feature vectors:
print("Concatenated shape:", concatenated_feature_vectors_su_train.shape)

concatenated_feature_vectors_su_validation = np.concatenate(
    (fft_su_validation_feature_vectors,
     resnet_su_validation_feature_vectors,
     rocket_su_validation_feature_vectors,
     shapelet_su_validation_feature_vectors),
    axis=1
)
print("Concatenated shape:", concatenated_feature_vectors_su_validation.shape)

####
concatenated_feature_vectors_su_test = np.concatenate(
    (fft_su_test_feature_vectors,
     resnet_su_test_feature_vectors,
     rocket_su_test_feature_vectors,
     shapelet_su_test_feature_vectors),
    axis=1
)

# Check the shape of the concatenated feature vectors:
print("Concatenated shape:", concatenated_feature_vectors_su_test.shape)

###
print("*** JIGSAWS Suturing CBR Feature Fusion ***  " + (args.validation_option) )


(x_train, y_train), (x_validation, y_validation), (x_test, y_test), (x_full_train, y_full_train, x_train_original, x_test_original, SU_LOSO_Full_Train) = load_data(args.validation_option)

scaler = MinMaxScaler()

fft_test_scaled = scaler.fit_transform(fft_su_test_feature_vectors)
resnet_test_scaled = scaler.fit_transform(resnet_su_test_feature_vectors)
rocket_test_scaled = scaler.fit_transform(rocket_su_test_feature_vectors)
shapelet_test_scaled = scaler.fit_transform(shapelet_su_test_feature_vectors)

concatenated_feature_vectors_su_test_scaled = np.concatenate(
    (fft_test_scaled, resnet_test_scaled, rocket_test_scaled, shapelet_test_scaled),
    axis=1
)

fft_validation_scaled = scaler.fit_transform(fft_su_validation_feature_vectors)
resnet_validation_scaled = scaler.fit_transform(resnet_su_validation_feature_vectors)
rocket_validation_scaled = scaler.fit_transform(rocket_su_validation_feature_vectors)
shapelet_validation_scaled = scaler.fit_transform(shapelet_su_validation_feature_vectors)

concatenated_feature_vectors_su_validation_scaled = np.concatenate(
    (fft_validation_scaled, resnet_validation_scaled, rocket_validation_scaled, shapelet_validation_scaled),
    axis=1
)


fft_train_scaled = scaler.fit_transform(fft_su_train_feature_vectors)
resnet_train_scaled = scaler.fit_transform(resnet_su_train_feature_vectors)
rocket_train_scaled = scaler.fit_transform(rocket_su_train_feature_vectors)
shapelet_train_scaled = scaler.fit_transform(shapelet_su_train_feature_vectors)

concatenated_feature_vectors_su_train_scaled = np.concatenate(
    (fft_train_scaled, resnet_train_scaled, rocket_train_scaled, shapelet_train_scaled),
    axis=1
)


####

# evaluate_test_vectors_to_csv(concatenated_feature_vectors_su_validation_scaled, concatenated_feature_vectors_su_train_scaled, y_test, y_full_train, './' + "CBR_su_feature_fusion_scaled_" + str(args.validation_option) + '_output.csv')

# predictions_1nn, counterfactuals_1nn, predictions_knn, predictions_1nn_trial_names, counterfactual_1nn_trial_names  = evaluate_test_vectors_with_counterfactuals(concatenated_feature_vectors_su_validation_scaled, concatenated_feature_vectors_su_train_scaled, y_test, y_full_train, SU_LOSO_Full_Train)

evaluate_test_vectors_to_csv(concatenated_feature_vectors_su_test_scaled, concatenated_feature_vectors_su_train_scaled, y_test, y_train, './' + "CBR_su_feature_fusion_scaled_" + str(args.validation_option) + '_output.csv')

predictions_1nn, counterfactuals_1nn, predictions_knn, predictions_1nn_trial_names, counterfactual_1nn_trial_names  = evaluate_test_vectors_with_counterfactuals(concatenated_feature_vectors_su_test_scaled, concatenated_feature_vectors_su_train_scaled, y_test, y_train, SU_LOSO_Full_Train)
###

# Predict on the test set
print("Prediction results 1-NN: ", predictions_1nn)
test_accuracy = accuracy_score(y_test, predictions_1nn)
print("Test accuracy 1-NN: ", test_accuracy)
test_accuracy_knn = accuracy_score(y_test, predictions_knn)
print("Test accuracy 3-NN: ", test_accuracy_knn)
test_precision = precision_score(y_test, predictions_1nn, average='macro', zero_division=0)
print("Test precision: ", test_precision)
test_recall = recall_score(y_test, predictions_1nn, average='macro', zero_division=0)
print("Test recall: ", test_recall)
test_f1 = f1_score(y_test, predictions_1nn, average='macro', zero_division=0)
print("Test F1: ", test_f1)
test_confusion_matrix = confusion_matrix(y_test, predictions_1nn)
print("Test confusion matrix: \n", test_confusion_matrix)

# Save the results to a csv file
results = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Test-Prediction(CBR-1NN)", "Test-Prediction(CBR-3NN)", "Factual-trial-names", "Counterfactual-trial-names"],
    "Value": [test_accuracy, test_precision, test_recall, test_f1, predictions_1nn, test_accuracy_knn, predictions_1nn_trial_names, counterfactual_1nn_trial_names]
}
df = pd.DataFrame(results)

####
file_path = './results/jigsaws/' + "CBR_su_feature_fusion_scaled_" + str(args.validation_option) + '_results.csv'
df.to_csv(file_path, index=False)

## append log outputs of full retrieval results

####
file_path_log = './' + "CBR_su_feature_fusion_scaled_" + str(args.validation_option) + '_output.csv'
df_log = pd.read_csv(file_path_log)

with open(file_path, 'a') as f:
    f.write('\n\n')  
    df_log.to_csv(f, index=False)