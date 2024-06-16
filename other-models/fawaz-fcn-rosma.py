import numpy as np
import random
#import imageio
import time
from itertools import chain
import os
import pandas as pd
import collections
import re
import math
import argparse
import datetime
import pytz
import matplotlib
import matplotlib.pyplot as plt
import io
from mpl_toolkits.mplot3d import Axes3D
import json
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers, models,Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
###
import sys
sys.path.append('./')
from datasets.rosma_ps import load_data

def reshape_data(input_data, input_shapes):
    reshaped_data = []
    for shapes in input_shapes:
        hand_data = []
        start_idx = 0
        for shape in shapes:
            end_idx = start_idx + shape[1]
            hand_data.append(input_data[:, :, start_idx:end_idx])
            start_idx = end_idx
        reshaped_data.append(hand_data)
    return reshaped_data


####
parser = argparse.ArgumentParser(description='ROSMA Post and sleeve Fawaz FCN')
parser.add_argument('--validation_option', type=str, default='LOSO1', help='Cross Validation Options')
parser.add_argument('--epochs', type=int, default=1000)
args = parser.parse_args()
###

####
(x_train, y_train), (x_validation, y_validation), (x_test, y_test), (x_full_train, y_full_train, x_train_original, x_test_original, PS_Full_Train) = load_data(args.validation_option)
print(PS_Full_Train)

# rosma, total 154 kinematic variables, MTML: 3,4,3,3,3,3,7,7,7 (40); MTMR: 3,4,3,3,3,3,7,7,7 (40); PSM1: 3,4,3,3,3,3,6,6,6 (37); PSM2: 3,4,3,3,3,3,6,6,6 (37); None (batch size)
input_shapes = [[(None,3),(None,4),(None,3),(None,3),(None,3),(None,3),(None,7),(None,7),(None,3)],  
                [(None,3),(None,4),(None,3),(None,3),(None,3),(None,3),(None,7),(None,7),(None,3)],  
                [(None,3),(None,4),(None,3),(None,3),(None,3),(None,3),(None,6),(None,6),(None,6)],  
                [(None,3),(None,9),(None,3),(None,3),(None,1),(None,3),(None,7),(None,6),(None,6)]]

n_classes = 2

random_indices = np.random.choice(53, 12, replace=False)
x_validation_original = x_train_original[random_indices, :, :]
y_validation = y_full_train[random_indices]

reshaped_x_full_train = reshape_data(x_train_original, input_shapes)
reshaped_x_validation = reshape_data(x_validation_original, input_shapes)
reshaped_x_test = reshape_data(x_test_original, input_shapes)

label_encoder = LabelEncoder()
y_train_int = label_encoder.fit_transform(y_full_train)
y_train_one_hot = to_categorical(y_train_int)

y_validation_int = label_encoder.fit_transform(y_validation)
y_validation_one_hot = to_categorical(y_validation_int)

y_test_int = label_encoder.fit_transform(y_test)
y_test_one_hot = to_categorical(y_test_int)

for i, group in enumerate(reshaped_x_test):
    print(f"Group {i}:")
    for j, df in enumerate(group):
        print(f" shape: {df.shape}")

def create_convolutional_network(input_shapes, num_filters_first_conv=8, num_filters_second_conv=16, 
                                 num_filters_final_conv=32, kernel_size=3, reg_strength=0.00001, 
                                 classes=[0, 1]):
    """
    Create a multi-input convolutional neural network.

    Parameters:
    - input_shapes: List of input shapes for each hand and each dimension cluster.
    - num_filters_first_conv, num_filters_second_conv, num_filters_final_conv: Number of filters for the conv layers.
    - kernel_size: Kernel size for convolution.
    - reg_strength: Regularization strength.
    - classes: Output classes.

    Returns:
    - model: Constructed Keras model.
    """

    def create_conv_layer(input_layer, num_filters, kernel_size, reg_strength):
        """
        Helper function to create a convolutional layer.
        """
        conv_layer = layers.Conv1D(num_filters, kernel_size=kernel_size, strides=1, padding='same', 
                                   activity_regularizer=regularizers.l2(reg_strength))(input_layer)
        conv_layer = layers.Activation('relu')(conv_layer)
        return conv_layer

    nb_classes = len(classes)
    num_hands = len(input_shapes)
    num_dim_clusters = len(input_shapes[0])

    # Initializing layers
    input_layers = [[None for _ in range(num_dim_clusters)] for _ in range(num_hands)]
    first_conv_layers = [[None for _ in range(num_dim_clusters)] for _ in range(num_hands)]
    hand_concatenated_layers = [None for _ in range(num_hands)]
    second_conv_layers = [None for _ in range(num_hands)]

    # Constructing the model
    for i in range(num_hands):
        for j in range(num_dim_clusters):
            input_layers[i][j] = layers.Input(shape=input_shapes[i][j])
            first_conv_layers[i][j] = create_conv_layer(input_layers[i][j], num_filters_first_conv, kernel_size, reg_strength)
        
        hand_concatenated_layers[i] = layers.Concatenate(axis=-1)(first_conv_layers[i])
        second_conv_layers[i] = create_conv_layer(hand_concatenated_layers[i], num_filters_second_conv, kernel_size, reg_strength)

    final_input = layers.Concatenate(axis=-1)(second_conv_layers)
    final_conv_layer = create_conv_layer(final_input, num_filters_final_conv, kernel_size, reg_strength)
    global_avg_pooling = layers.GlobalAveragePooling1D()(final_conv_layer)
    output_layer = layers.Dense(nb_classes, activation='softmax')(global_avg_pooling)

    # Create the model by linking input and output
    model_inputs = list(chain.from_iterable(input_layers))
    model = models.Model(inputs=model_inputs, outputs=output_layer)

    return model

model = create_convolutional_network(input_shapes)
model.summary()

model.compile(optimizer='adam',  
              loss='categorical_crossentropy',  
              metrics=['accuracy'])

history = model.fit(reshaped_x_full_train, y_train_one_hot, batch_size=32, epochs=args.epochs)

val_accuracy = model.evaluate(reshaped_x_validation, y_validation_one_hot)
print("Validation Accuracy:", val_accuracy[1])

#accuracy = model.evaluate(reshaped_x_test, y_test_one_hot)
#print("Test Accuracy:", accuracy)

# Predict on the test set
test_pred = model.predict(reshaped_x_test)
print("Prediction: ", test_pred)
predicted_classes = np.argmax(test_pred, axis=1)

y_test_labels = np.argmax(y_test_one_hot, axis=1)
print("Predicted class labels:",predicted_classes)

accuracy = accuracy_score(y_test_labels, predicted_classes)
print("Test accuracy: ", accuracy)

test_precision = precision_score(y_test_labels, predicted_classes, average='macro', zero_division=0)
print("Test precision: ", test_precision)
test_recall = recall_score(y_test_labels, predicted_classes, average='macro', zero_division=0)
print("Test recall: ", test_recall)
test_f1 = f1_score(y_test_labels, predicted_classes, average='macro', zero_division=0)
print("Test F1: ", test_f1)
test_confusion_matrix = confusion_matrix(y_test_labels, predicted_classes)
print("Test confusion matrix: \n", test_confusion_matrix)

#roc_auc = roc_auc_score(y_test_one_hot, test_pred)
#print("Test ROC AUC: ", roc_auc)
unique_values = np.unique(y_test_one_hot)
if len(unique_values) == 1:
    roc_auc = "None"
else:
    roc_auc = roc_auc_score(y_test_one_hot, test_pred)
print("Test ROC AUC: ", roc_auc)

results = {
    "Metric": ["Validation-Accuracy", "Test-Accuracy", "Test-Precision", "Test-Recall", "Test-F1-Score", "Test-Prediction", "Test-AUC"],
    "Value": [val_accuracy[1], accuracy, test_precision, test_recall, test_f1, predicted_classes, roc_auc]
}
df = pd.DataFrame(results)

#####
file_path = './IJCAI-24/src/results/rosma/' + "rosmaFCN_ps_" + str(args.validation_option) + '_results.csv'
df.to_csv(file_path, index=False)