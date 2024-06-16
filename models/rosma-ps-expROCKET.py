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
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.calibration import CalibratedClassifierCV

####
parser = argparse.ArgumentParser(description='ROSMA Post and Sleeve ROCKET')
parser.add_argument('--validation_option', type=str, default='LOSO1', help='Cross Validation Options')
args = parser.parse_args()

print("*** ROSMA Post and Sleeve ROCKET ***  " + (args.validation_option) )


####
(x_train, y_train), (x_validation, y_validation), (x_test, y_test), (x_full_train, y_full_train, x_train_original, x_test_original, PS_Full_Train) = load_data(args.validation_option)

print("Train shape:  ", x_train.shape)
print("Validation shape:  ", x_validation.shape)
print("Test shape:  ", x_test.shape)

print(y_train)
print(y_validation)
print(y_test)


# Create the pipeline with Rocket and RidgeClassifierCV
# rocket_pipeline = make_pipeline(Rocket(), RidgeClassifierCV())

# Fit the pipeline
# rocket_pipeline.fit(x_train, y_train)

# # Transform the training data with all but the last step of the pipeline (i.e., Rocket)
# transformed_x_train = rocket_pipeline[:-1].transform(x_train)

# # Extract the RidgeClassifierCV instance and fit it on the transformed data
# ridge_classifier = rocket_pipeline.named_steps['ridgeclassifiercv']
# ridge_classifier.fit(transformed_x_train, y_train)

# # Now use CalibratedClassifierCV with the fitted RidgeClassifierCV
# calibrated_clf = CalibratedClassifierCV(base_estimator=ridge_classifier, cv='prefit')
# calibrated_clf.fit(transformed_x_train, y_train)

# probabilities = calibrated_clf.predict_proba(rocket_pipeline[:-1].transform(x_test))
# # Getting the predictions
# predictions = rocket_pipeline.predict(x_test)


# # Print probabilities and predictions
# print("Probabilities (Logits):", probabilities)
# print("Predictions:", predictions)


# validation_option = str(args.validation_option) 

# formatted_probabilities = [f"[{prob[0]} {prob[1]}]" for prob in probabilities]
# df = pd.DataFrame({'Probabilities': formatted_probabilities,
#                    'Predictions': predictions})


# file_name = f"rocket_ps_multimodality_{validation_option}.csv"
# file_path = f"./results/rosma/{file_name}"

# df.to_csv(file_path, index=False)

# print(f"Saved to {file_path}")



rocket_pipeline = make_pipeline(Rocket(), RidgeClassifierCV())
rocket_pipeline.fit(x_train, y_train)

# Predict on the test set
test_pred = rocket_pipeline.predict(x_test)
print("Prediction: ", test_pred)
test_accuracy = accuracy_score(y_test, test_pred)
print("Test accuracy: ", test_accuracy)
test_precision = precision_score(y_test, test_pred, average='macro', zero_division=0)
print("Test precision: ", test_precision)
test_recall = recall_score(y_test, test_pred, average='macro', zero_division=0)
print("Test recall: ", test_recall)
test_f1 = f1_score(y_test, test_pred, average='macro', zero_division=0)
print("Test F1: ", test_f1)
test_confusion_matrix = confusion_matrix(y_test, test_pred)
print("Test confusion matrix: \n", test_confusion_matrix)

# Save the results to a csv file

results = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Test-Prediction"],
    "Value": [test_accuracy, test_precision, test_recall, test_f1, test_pred]
}
df = pd.DataFrame(results)

####
file_path = './results/rosma/' + "rocket_ps_" + str(args.validation_option) + '_results.csv'
df.to_csv(file_path, index=False)

trf = Rocket(num_kernels=512) 
trf.fit(x_train) 

train_transformed = trf.transform(x_train) 
validation_transformed = trf.transform(x_validation) 
test_transformed = trf.transform(x_test) 

print("Train feature vectors shape: ", train_transformed.shape)
print("Validation feature vectors shape: ", validation_transformed.shape)
print("Test feature vectors shape: ", test_transformed.shape)

####
np.save('./results/rosma/' + "./rocket_ps_" + str(args.validation_option) + "_train_feature_vectors.npy", train_transformed)
np.save('./results/rosma/' + "./rocket_ps_" + str(args.validation_option) + "_validation_feature_vectors.npy", validation_transformed)
np.save('./results/rosma/' + "./rocket_ps_" + str(args.validation_option) + "_test_feature_vectors.npy", test_transformed)

