"""
ROSMA wire chaser dataset
"""
import os 
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
import pickle
import pandas as pd
import random
from sktime.datatypes import convert
from pathlib import Path

def get_LOSO_sets(data_list, loso_suffix):
    test_set = [s for s in data_list if s.endswith(loso_suffix)]
    train_set = [s for s in data_list if not s.endswith(loso_suffix)]
    return test_set, train_set

def rosma_get_LOUO_sets(data_list, louo_suffix): # X12
    test_set = [s for s in data_list if s.startswith(louo_suffix)]
    train_set = [s for s in data_list if not s.startswith(louo_suffix)]
    return test_set, train_set

def select_validation_set(train_set, num_trials=12): # change to 12 for ROSMA
    validation_set = random.sample(train_set, num_trials)
    remaining_train_set = [item for item in train_set if item not in validation_set]
    return validation_set, remaining_train_set

## Preprocessing: padding
def padding_wrap(samples):
    lengths = [len(i) for i in samples]
    print("Length of the input: " + str(lengths))
    max_len = max(lengths)
    print("Max length is: "+ str(max_len))
    for i in range(len(samples)):
        pad_len = max_len - lengths[i]
        samples[i] = np.pad(samples[i],((0,pad_len),(0,0)),"wrap")
    return np.stack(samples)


def load_data(fold_number):
   
    """
    Load and return the ROSMA Wire chaser dataset.
    ## fold_number: "LOSO1"
    ==============             ==============
    Total Trials                           71
    Training Samples total                 59
    Testing Samples total                  12 
    Number of time steps                  3853
    Dimensionality                         154
    Number of targets                       2
    ==============             ==============

    # Returns
        Numpy arrays: (x_train, y_train), (x_test, y_test)
    """

    module_path = os.getcwd()

    ## please visit the offical dataset website of rosma to download, and organise it into df_dataset_wc.pkl for wire chaser
    ## the annotation of rosma will be public within the acceptance of the paper
    df_dataset_wc = pd.read_pickle(module_path + '/datasets/rosma-kinematic/df_dataset_wc.pkl')

    kinemtiac_raw_data_wire_chaser = df_dataset_wc['kinematic_data'].to_numpy()
    padding_data_wire_chaser = padding_wrap(kinemtiac_raw_data_wire_chaser)

    reshaped_data_wc_padding = padding_data_wire_chaser.transpose(0, 2, 1)
    padding_data_wire_chaser_mtype = convert(reshaped_data_wc_padding, from_type="numpy3D", to_type="pd-multiindex")
    padding_data_wire_chaser_nested_univ = convert(padding_data_wire_chaser_mtype, from_type="pd-multiindex", to_type="nested_univ")      
    
    wire_chaser_trial_name = df_dataset_wc['Trial'].to_numpy()

    if not 'X' in fold_number: ## LOSO1
        print("************ LOSO Cross Validation ************")

        loso_suffix = "0" + fold_number[-1]  ### different JIGSAWS, ROSMA name ends with one '0', e.g.: X12_Pea_on_a_Peg_06
        WC_LOSO_Test, WC_Full_Train = get_LOSO_sets(wire_chaser_trial_name, loso_suffix)
        WC_LOSO_Validation, WC_LOSO_Train = select_validation_set(WC_Full_Train)

        print("WC LOSO Test Trial Names: ", WC_LOSO_Full_Train)

        indices_wc_loso_test = [np.where(wire_chaser_trial_name == a)[0][0] for a in WC_LOSO_Test]
        indices_wc_loso_full_train = [np.where(wire_chaser_trial_name == a)[0][0] for a in WC_Full_Train]  

        indices_wc_loso_train = [np.where(wire_chaser_trial_name == a)[0][0] for a in WC_LOSO_Train]
        indices_wc_loso_validation = [np.where(wire_chaser_trial_name == a)[0][0] for a in WC_LOSO_Validation]
        
        wc_loso_train = padding_data_wire_chaser_nested_univ.iloc[indices_wc_loso_train]
        wc_loso_validation = padding_data_wire_chaser_nested_univ.iloc[indices_wc_loso_validation]

        wc_loso_full_train = padding_data_wire_chaser_nested_univ.iloc[indices_wc_loso_full_train]
        wc_loso_test = padding_data_wire_chaser_nested_univ.iloc[indices_wc_loso_test]

        ## in original shape
        wc_loso_train_original = padding_data_wire_chaser[indices_wc_loso_full_train, :, :]
        wc_loso_test_original = padding_data_wire_chaser[indices_wc_loso_test, :, :]

        ## label
        wire_chaser_label = df_dataset_wc['skill_level'].to_numpy()

        wc_loso_train_labels = wire_chaser_label[indices_wc_loso_train]
        wc_loso_full_train_labels = wire_chaser_label[indices_wc_loso_full_train]
        wc_loso_validation_labels = wire_chaser_label[indices_wc_loso_validation]
        wc_loso_test_labels = wire_chaser_label[indices_wc_loso_test]

        x_train = wc_loso_train
        x_validation = wc_loso_validation

        x_full_train = wc_loso_full_train
        x_test = wc_loso_test

        x_train_original = wc_loso_train_original
        x_test_original = wc_loso_test_original

        y_train = wc_loso_train_labels
        y_validation = wc_loso_validation_labels
        y_test = wc_loso_test_labels
        y_full_train = wc_loso_full_train_labels

    else:
        print("************ LOUO Cross Validation ************")

        louo_suffix = fold_number[-3:]  # X01, X12
        WC_LOUO_Test, WC_Full_Train = rosma_get_LOUO_sets(wire_chaser_trial_name, louo_suffix)

        WC_LOUO_Validation, WC_LOUO_Train = select_validation_set(WC_Full_Train)

        print("WC LOUO Test Trial Names: ", WC_LOUO_Test)
        
        indices_wc_louo_test = [np.where(wire_chaser_trial_name == a)[0][0] for a in WC_LOUO_Test]
        indices_wc_louo_full_train = [np.where(wire_chaser_trial_name == a)[0][0] for a in WC_Full_Train]  

        indices_wc_louo_train = [np.where(wire_chaser_trial_name == a)[0][0] for a in WC_LOUO_Train]
        indices_wc_louo_validation = [np.where(wire_chaser_trial_name == a)[0][0] for a in WC_LOUO_Validation]
        
        wc_louo_train = padding_data_wire_chaser_nested_univ.iloc[indices_wc_louo_train]
        wc_louo_validation = padding_data_wire_chaser_nested_univ.iloc[indices_wc_louo_validation]

        wc_louo_full_train = padding_data_wire_chaser_nested_univ.iloc[indices_wc_louo_full_train]
        wc_louo_test = padding_data_wire_chaser_nested_univ.iloc[indices_wc_louo_test]

        ## in original shape
        wc_louo_train_original = padding_data_wire_chaser[indices_wc_louo_full_train, :, :]
        wc_louo_test_original = padding_data_wire_chaser[indices_wc_louo_test, :, :]

        ## label
        wire_chaser_label = df_dataset_wc['skill_level'].to_numpy()

        wc_louo_train_labels = wire_chaser_label[indices_wc_louo_train]
        wc_louo_full_train_labels = wire_chaser_label[indices_wc_louo_full_train]
        wc_louo_validation_labels = wire_chaser_label[indices_wc_louo_validation]
        wc_louo_test_labels = wire_chaser_label[indices_wc_louo_test]

        x_train = wc_louo_train
        x_validation = wc_louo_validation

        x_full_train = wc_louo_full_train
        x_test = wc_louo_test

        x_train_original = wc_louo_train_original
        x_test_original = wc_louo_test_original

        y_train = wc_louo_train_labels
        y_validation = wc_louo_validation_labels
        y_test = wc_louo_test_labels
        y_full_train = wc_louo_full_train_labels        

    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test), (x_full_train, y_full_train, x_train_original, x_test_original, WC_Full_Train)