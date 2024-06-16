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

###
parser = argparse.ArgumentParser(description='ROSMA Post and sleeve ResNet')
parser.add_argument('--validation_option', type=str, default='LOSO1', help='Cross Validation Options')
parser.add_argument('--epochs', type=int, default=500)
args = parser.parse_args()

###
print("*** ROSMA Post and sleeve ResNet ***  " + (args.validation_option) )

###
(x_train, y_train), (x_validation, y_validation), (x_test, y_test), (x_full_train, y_full_train, x_train_original, x_test_original, PS_Full_Train) = load_data(args.validation_option)

print("Train shape:  ", x_train.shape)
print("Validation shape:  ", x_validation.shape)
print("Test shape:  ", x_test.shape)

print(y_train)
print(y_validation)
print(y_test)

resnet = ResNetClassifier(n_epochs= args.epochs, verbose=True)
resnet.fit(x_train, y_train)

# extract from the last second layer 
feature_extractor_model = Model(inputs=resnet.model_.input, outputs=resnet.model_.layers[-2].output)

#####             
resnet_ps_train_feature_vectors = feature_extractor_model.predict(x_train)
resnet_ps_validation_feature_vectors = feature_extractor_model.predict(x_validation)
resnet_ps_test_feature_vectors = feature_extractor_model.predict(x_test) 


# Predict on the test set
prob_predictions = resnet.predict_proba(x_test)
positive_prob_predictions = prob_predictions[:, 1]

try:
    test_auc = roc_auc_score(y_test, positive_prob_predictions)
    print("AUC: ", test_auc)
except ValueError as e:
    print("Error computing ROC AUC: ", e)
    test_auc = 0
test_pred = resnet.predict(x_test)
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
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Test-Prediction", "AUC"],
    "Value": [test_accuracy, test_precision, test_recall, test_f1, test_pred, test_auc]
}
df = pd.DataFrame(results)

####
file_path = './results/rosma/' + "resnet_ps_" + str(args.validation_option) + '_results.csv'
df.to_csv(file_path, index=False)

####
np.save('./results/rosma/' + "./resnet_ps_" + str(args.validation_option) + "_train_feature_vectors.npy", resnet_ps_train_feature_vectors)
np.save('./results/rosma/' + "./resnet_ps_" + str(args.validation_option) + "_validation_feature_vectors.npy", resnet_ps_validation_feature_vectors)
np.save('./results/rosma/' + "./resnet_ps_" + str(args.validation_option) + "_test_feature_vectors.npy", resnet_ps_test_feature_vectors)

