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
from datasets.jigsaws_np import load_data
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
#from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform

parser = argparse.ArgumentParser(description='JIGSAWS Needle passing Shapelet')
parser.add_argument('--validation_option', type=str, default='LOSO1', help='Cross Validation Options')
args = parser.parse_args()

print("*** JIGSAWS Needle passing Shapelet ***  " + (args.validation_option) )

(x_train, y_train), (x_validation, y_validation), (x_test, y_test), (x_full_train, y_full_train, x_train_original, x_test_original, NP_Full_Train) = load_data(args.validation_option)

print("Train shape:  ", x_train.shape)
print("Validation shape:  ", x_validation.shape)
print("Test shape:  ", x_test.shape)

print(y_train)
print(y_validation)
print(y_test)

# su channel selection: based on the results of su loso 1: select useful channels from 76 channels
#su_cs_index = [17,18, 22, 25, 26, 28, 29, 35, 36, 37, 41, 44, 45, 47, 48, 54, 55, 60, 61, 63, 64, 66, 67, 74]
#su_cs_index_str = ['var_' + str(i) for i in su_cs_index]

#su_train_cs = x_full_train.loc[:, su_cs_index_str]
#su_test_cs = x_test.loc[:, su_cs_index_str]

# Initialize and train the ShapeletTransformClassifier: shapelet number 50, 60, 70, 80 (500)
stc = RandomShapeletTransform(n_shapelet_samples=500, max_shapelets=70)
stc.fit(x_train, y_train)

train_transformed = stc.transform(x_train)
validation_transformed = stc.transform(x_validation)
test_transformed = stc.transform(x_test)

print("Train feature vectors shape: ", train_transformed.shape)
print("Validation feature vectors shape: ", validation_transformed.shape)
print("Test feature vectors shape: ", test_transformed.shape)

np.save('./results/jigsaws/' + "./shapelet_np_" + str(args.validation_option) + "_train_feature_vectors.npy", train_transformed)
np.save('./results/jigsaws/' + "./shapelet_np_" + str(args.validation_option) + "_validation_feature_vectors.npy", validation_transformed)
np.save('./results/jigsaws/' + "./shapelet_np_" + str(args.validation_option) + "_test_feature_vectors.npy", test_transformed)

