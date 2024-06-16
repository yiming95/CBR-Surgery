"""
ROSMA Pea on a peg dataset
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

# Convert to 3D array and pad
#pp_array_3d = np.array([df.to_numpy() for df in pp_dataframe])
#padding_data_pp = padding_wrap(pp_array_3d)
#reshaped_data_pp_padding = padding_data_pp.transpose(0, 2, 1)

# Conversion (ensure convert function is defined)
#pp_mtype = convert(reshaped_data_pp_padding, from_type="numpy3D", to_type="pd-multiindex")
#pp_nested_univ = convert(pp_mtype, from_type="pd-multiindex", to_type="nested_univ")

# Print shapes for verification
#print(padding_data_pp.shape)
#print(reshaped_data_pp_padding.shape)
#print(pp_mtype.shape)
#print(pp_nested_univ.shape)



def load_data(fold_number):
   
    """
    Load and return the ROSMA Pea on a peg dataset.
    ## fold_number: "LOSO1"
    ==============             ==============
    Total Trials                           70
    Training Samples total                 58
    Testing Samples total                  12 
    Number of time steps                  3853
    Dimensionality                         154
    Number of targets                       2
    ==============             ==============

    """

    module_path = os.getcwd()

    # df_dataset_pp, df_dataset_ps, df_dataset_wc

    ## please visit the offical dataset website of rosma to download, and organise it into df_dataset_pp.pkl for pea on a peg
    ## the annotation of rosma will be public within the acceptance of the paper
    df_dataset_pp = pd.read_pickle(module_path + '/datasets/rosma-kinematic/df_dataset_pp.pkl')

    kinemtiac_raw_data_pea_on_a_peg = df_dataset_pp['kinematic_data'].to_numpy()
    padding_data_pea_on_a_peg = padding_wrap(kinemtiac_raw_data_pea_on_a_peg)

    reshaped_data_pp_padding = padding_data_pea_on_a_peg.transpose(0, 2, 1)
    padding_data_pea_on_a_peg_mtype = convert(reshaped_data_pp_padding, from_type="numpy3D", to_type="pd-multiindex")
    padding_data_pea_on_a_peg_nested_univ = convert(padding_data_pea_on_a_peg_mtype, from_type="pd-multiindex", to_type="nested_univ")      
    
    pea_on_a_peg_trial_name = df_dataset_pp['Trial'].to_numpy()

    if not 'X' in fold_number: ## LOSO1
        print("************ LOSO Cross Validation ************")

        loso_suffix = "0" + fold_number[-1]  ### different JIGSAWS, ROSMA name ends with one '0', e.g.: X12_Pea_on_a_Peg_06
        PP_LOSO_Test, PP_Full_Train = get_LOSO_sets(pea_on_a_peg_trial_name, loso_suffix)
        PP_LOSO_Validation, PP_LOSO_Train = select_validation_set(PP_Full_Train)

        print("PP Test Trial Names: ", PP_LOSO_Test)
        
        indices_pp_loso_test = [np.where(pea_on_a_peg_trial_name == a)[0][0] for a in PP_LOSO_Test]
        indices_pp_loso_full_train = [np.where(pea_on_a_peg_trial_name == a)[0][0] for a in PP_Full_Train]  

        indices_pp_loso_train = [np.where(pea_on_a_peg_trial_name == a)[0][0] for a in PP_LOSO_Train]
        indices_pp_loso_validation = [np.where(pea_on_a_peg_trial_name == a)[0][0] for a in PP_LOSO_Validation]
        
        pp_loso_train = padding_data_pea_on_a_peg_nested_univ.iloc[indices_pp_loso_train]
        pp_loso_validation = padding_data_pea_on_a_peg_nested_univ.iloc[indices_pp_loso_validation]

        pp_loso_full_train = padding_data_pea_on_a_peg_nested_univ.iloc[indices_pp_loso_full_train]
        pp_loso_test = padding_data_pea_on_a_peg_nested_univ.iloc[indices_pp_loso_test]

        ## in original shape
        pp_loso_train_original = padding_data_pea_on_a_peg[indices_pp_loso_full_train, :, :]
        pp_loso_test_original = padding_data_pea_on_a_peg[indices_pp_loso_test, :, :]

        ## label
        pea_on_a_peg_label = df_dataset_pp['skill_level'].to_numpy()

        pp_loso_train_labels = pea_on_a_peg_label[indices_pp_loso_train]
        pp_loso_full_train_labels = pea_on_a_peg_label[indices_pp_loso_full_train]
        pp_loso_validation_labels = pea_on_a_peg_label[indices_pp_loso_validation]
        pp_loso_test_labels = pea_on_a_peg_label[indices_pp_loso_test]

        x_train = pp_loso_train
        x_validation = pp_loso_validation

        x_full_train = pp_loso_full_train
        x_test = pp_loso_test

        x_train_original = pp_loso_train_original
        x_test_original = pp_loso_test_original

        y_train = pp_loso_train_labels
        y_validation = pp_loso_validation_labels
        y_test = pp_loso_test_labels
        y_full_train = pp_loso_full_train_labels

    else:
        print("************ LOUO Cross Validation ************")

        louo_suffix = fold_number[-3:]  # X01, X12
        PP_LOUO_Test, PP_Full_Train = rosma_get_LOUO_sets(pea_on_a_peg_trial_name, louo_suffix)

        PP_LOUO_Validation, PP_LOUO_Train = select_validation_set(PP_Full_Train)

        print("PP LOUO Test Trial Names: ", PP_LOUO_Test)
        
        indices_pp_louo_test = [np.where(pea_on_a_peg_trial_name == a)[0][0] for a in PP_LOUO_Test]
        indices_pp_louo_full_train = [np.where(pea_on_a_peg_trial_name == a)[0][0] for a in PP_Full_Train]  

        indices_pp_louo_train = [np.where(pea_on_a_peg_trial_name == a)[0][0] for a in PP_LOUO_Train]
        indices_pp_louo_validation = [np.where(pea_on_a_peg_trial_name == a)[0][0] for a in PP_LOUO_Validation]
        
        pp_louo_train = padding_data_pea_on_a_peg_nested_univ.iloc[indices_pp_louo_train]
        pp_louo_validation = padding_data_pea_on_a_peg_nested_univ.iloc[indices_pp_louo_validation]

        pp_louo_full_train = padding_data_pea_on_a_peg_nested_univ.iloc[indices_pp_louo_full_train]
        pp_louo_test = padding_data_pea_on_a_peg_nested_univ.iloc[indices_pp_louo_test]

        ## in original shape
        pp_louo_train_original = padding_data_pea_on_a_peg[indices_pp_louo_full_train, :, :]
        pp_louo_test_original = padding_data_pea_on_a_peg[indices_pp_louo_test, :, :]

        ## label
        pea_on_a_peg_label = df_dataset_pp['skill_level'].to_numpy()

        pp_louo_train_labels = pea_on_a_peg_label[indices_pp_louo_train]
        pp_louo_full_train_labels = pea_on_a_peg_label[indices_pp_louo_full_train]
        pp_louo_validation_labels = pea_on_a_peg_label[indices_pp_louo_validation]
        pp_louo_test_labels = pea_on_a_peg_label[indices_pp_louo_test]

        x_train = pp_louo_train
        x_validation = pp_louo_validation

        x_full_train = pp_louo_full_train
        x_test = pp_louo_test

        x_train_original = pp_louo_train_original
        x_test_original = pp_louo_test_original

        y_train = pp_louo_train_labels
        y_validation = pp_louo_validation_labels
        y_test = pp_louo_test_labels
        y_full_train = pp_louo_full_train_labels


    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test), (x_full_train, y_full_train, x_train_original, x_test_original, PP_Full_Train)