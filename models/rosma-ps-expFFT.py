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
####
from datasets.rosma_ps import load_data
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
from sklearn.metrics import roc_auc_score

####
parser = argparse.ArgumentParser(description='ROSMA Post and Sleeve FFT')
parser.add_argument('--validation_option', type=str, default='LOSO1', help='Cross Validation Options')
args = parser.parse_args()

####
print("*** ROSMA Post and Sleeve FFT ***  " + (args.validation_option) )

####
(x_train, y_train), (x_validation, y_validation), (x_test, y_test), (x_full_train, y_full_train, x_train_original, x_test_original, PS_Full_Train) = load_data(args.validation_option)


print("Train shape:  ", x_train.shape)
print("Validation shape:  ", x_validation.shape)
print("Test shape:  ", x_test.shape)


print(y_train)
print(y_validation)
print(y_test)


fc_parameters = ComprehensiveFCParameters()

fourier_parameters = {key: value for key, value in fc_parameters.items() if 'fourier' in key}

fft_feature_extractor = TSFreshFeatureExtractor(
    default_fc_parameters=fourier_parameters
)

# Initialize the encoder
encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_validation_encoded = encoder.fit_transform(y_validation)
y_test_encoded = encoder.fit_transform(y_test)

# Create and fit the pipeline with the FFT feature extractor and the XGBClassifier
pipeline = make_pipeline(fft_feature_extractor, XGBClassifier())
pipeline.fit(x_train, y_train_encoded)


# Extract FFT features from the training data
fft_train_feature_vectors = pipeline.named_steps['tsfreshfeatureextractor'].transform(x_train)

fft_validation_feature_vectors = pipeline.named_steps['tsfreshfeatureextractor'].transform(x_validation)
# Extract FFT features from the test data
fft_test_feature_vectors = pipeline.named_steps['tsfreshfeatureextractor'].transform(x_test)

#validation_pred = resnet.predict(x_validation)
#val_accuracy = accuracy_score(y_validation, validation_pred)
#print("Validation accuracy: ", val_accuracy)
#val_precision = precision_score(y_validation, validation_pred, average='macro', zero_division=0)
#print("Validation precision:  ", val_precision)
#val_recall = recall_score(y_validation, validation_pred, average='macro', zero_division=0)
#print("Validation recall:  ", val_precision)
#val_f1 = f1_score(y_validation, validation_pred, average='macro', zero_division=0)
#print("Validation F1:  ", val_f1)
#val_confusion_matrix = confusion_matrix(y_validation, validation_pred)
#print("Validation confusion matrix: \n", val_confusion_matrix)

# Predict on the test set
prob_predictions = pipeline.predict_proba(x_test)
positive_prob_predictions = prob_predictions[:, 1]
try:
    test_auc = roc_auc_score(y_test, positive_prob_predictions)
    print("AUC: ", test_auc)
except ValueError as e:
    print("Error computing ROC AUC: ", e)
    test_auc = 0
test_pred = pipeline.predict(x_test)
print("Prediction: ", test_pred)
test_accuracy = accuracy_score(y_test_encoded, test_pred)
print("Test accuracy: ", test_accuracy)
test_precision = precision_score(y_test_encoded, test_pred, average='macro', zero_division=0)
print("Test precision: ", test_precision)
test_recall = recall_score(y_test_encoded, test_pred, average='macro', zero_division=0)
print("Test recall: ", test_recall)
test_f1 = f1_score(y_test_encoded, test_pred, average='macro', zero_division=0)
print("Test F1: ", test_f1)
test_confusion_matrix = confusion_matrix(y_test_encoded, test_pred)
print("Test confusion matrix: \n", test_confusion_matrix)

# Save the results to a csv file

results = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Test-Prediction", "AUC"],
    "Value": [test_accuracy, test_precision, test_recall, test_f1, test_pred, test_auc]
}
df = pd.DataFrame(results)

####
file_path = './results/rosma/' + "fft_ps_" + str(args.validation_option) + '_results.csv'
df.to_csv(file_path, index=False)

####
np.save('./results/rosma/' + "./fft_ps_" + str(args.validation_option) + "_train_feature_vectors.npy", fft_train_feature_vectors)
np.save('./results/rosma/' + "./fft_ps_" + str(args.validation_option) + "_validation_feature_vectors.npy", fft_validation_feature_vectors)
np.save('./results/rosma/' + "./fft_ps_" + str(args.validation_option) + "_test_feature_vectors.npy", fft_test_feature_vectors)

