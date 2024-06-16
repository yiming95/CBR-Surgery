"""
ROSMA Post and sleeve dataset
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
    Load and return the ROSMA Post and sleeve dataset.
    ## fold_number: "LOSO1"
    ==============             ==============
    Total Trials                           65
    Training Samples total                 53
    Testing Samples total                  12 
    Number of time steps                  3853
    Dimensionality                         154
    Number of targets                       2
    ==============             ==============

    """

    module_path = os.getcwd()

    ## please visit the offical dataset website of rosma to download, and organise it into df_dataset_ps.pkl for post and sleeve
    ## the annotation of rosma will be public within the acceptance of the paper
    df_dataset_ps = pd.read_pickle(module_path + '/datasets/rosma-kinematic/df_dataset_ps.pkl')

    kinemtiac_raw_data_post_and_sleeve = df_dataset_ps['kinematic_data'].to_numpy()
    padding_data_post_and_sleeve = padding_wrap(kinemtiac_raw_data_post_and_sleeve)

    reshaped_data_ps_padding = padding_data_post_and_sleeve.transpose(0, 2, 1)
    padding_data_post_and_sleeve_mtype = convert(reshaped_data_ps_padding, from_type="numpy3D", to_type="pd-multiindex")
    padding_data_post_and_sleeve_nested_univ = convert(padding_data_post_and_sleeve_mtype, from_type="pd-multiindex", to_type="nested_univ")      
    
    post_and_sleeve_trial_name = df_dataset_ps['Trial'].to_numpy()

    if not 'X' in fold_number: ## LOSO1

        print("************ LOSO Cross Validation ************")

        loso_suffix = "0" + fold_number[-1]  ### different JIGSAWS, ROSMA name ends with one '0', e.g.: X12_Pea_on_a_Peg_06
        PS_LOSO_Test, PS_Full_Train = get_LOSO_sets(post_and_sleeve_trial_name, loso_suffix)
        PS_LOSO_Validation, PS_LOSO_Train = select_validation_set(PS_Full_Train)

        print("PS LOSO Test Trial Names: ", PS_LOSO_Test)
        # print("PS Full Train Trial Names: ", PP_LOSO_Full_Train)

        indices_ps_loso_test = [np.where(post_and_sleeve_trial_name == a)[0][0] for a in PS_LOSO_Test]
        indices_ps_loso_full_train = [np.where(post_and_sleeve_trial_name == a)[0][0] for a in PS_Full_Train]  

        indices_ps_loso_train = [np.where(post_and_sleeve_trial_name == a)[0][0] for a in PS_LOSO_Train]
        indices_ps_loso_validation = [np.where(post_and_sleeve_trial_name == a)[0][0] for a in PS_LOSO_Validation]
        
        ps_loso_train = padding_data_post_and_sleeve_nested_univ.iloc[indices_ps_loso_train]
        ps_loso_validation = padding_data_post_and_sleeve_nested_univ.iloc[indices_ps_loso_validation]

        ps_loso_full_train = padding_data_post_and_sleeve_nested_univ.iloc[indices_ps_loso_full_train]
        ps_loso_test = padding_data_post_and_sleeve_nested_univ.iloc[indices_ps_loso_test]

        ## in original shape
        ps_loso_train_original = padding_data_post_and_sleeve[indices_ps_loso_full_train, :, :]
        ps_loso_test_original = padding_data_post_and_sleeve[indices_ps_loso_test, :, :]

        ## label
        post_and_sleeve_label = df_dataset_ps['skill_level'].to_numpy()

        ps_loso_train_labels = post_and_sleeve_label[indices_ps_loso_train]
        ps_loso_full_train_labels = post_and_sleeve_label[indices_ps_loso_full_train]
        ps_loso_validation_labels = post_and_sleeve_label[indices_ps_loso_validation]
        ps_loso_test_labels = post_and_sleeve_label[indices_ps_loso_test]

        x_train = ps_loso_train
        x_validation = ps_loso_validation

        x_full_train = ps_loso_full_train
        x_test = ps_loso_test

        x_train_original = ps_loso_train_original
        x_test_original = ps_loso_test_original

        y_train = ps_loso_train_labels
        y_validation = ps_loso_validation_labels
        y_test = ps_loso_test_labels
        y_full_train = ps_loso_full_train_labels

    else:
        print("************ LOUO Cross Validation ************")

        louo_suffix = fold_number[-3:]  # X01, X12
        PS_LOUO_Test, PS_Full_Train = rosma_get_LOUO_sets(post_and_sleeve_trial_name, louo_suffix)

        PS_LOUO_Validation, PS_LOUO_Train = select_validation_set(PS_Full_Train)

        print("PS LOUO Test Trial Names: ", PS_LOUO_Test)
        
        indices_ps_louo_test = [np.where(post_and_sleeve_trial_name == a)[0][0] for a in PS_LOUO_Test]
        indices_ps_louo_full_train = [np.where(post_and_sleeve_trial_name == a)[0][0] for a in PS_Full_Train]  

        indices_ps_louo_train = [np.where(post_and_sleeve_trial_name == a)[0][0] for a in PS_LOUO_Train]
        indices_ps_louo_validation = [np.where(post_and_sleeve_trial_name == a)[0][0] for a in PS_LOUO_Validation]
        
        ps_louo_train = padding_data_post_and_sleeve_nested_univ.iloc[indices_ps_louo_train]
        ps_louo_validation = padding_data_post_and_sleeve_nested_univ.iloc[indices_ps_louo_validation]

        ps_louo_full_train = padding_data_post_and_sleeve_nested_univ.iloc[indices_ps_louo_full_train]
        ps_louo_test = padding_data_post_and_sleeve_nested_univ.iloc[indices_ps_louo_test]

        ## in original shape
        ps_louo_train_original = padding_data_post_and_sleeve[indices_ps_louo_full_train, :, :]
        ps_louo_test_original = padding_data_post_and_sleeve[indices_ps_louo_test, :, :]

        ## label
        post_and_sleeve_label = df_dataset_ps['skill_level'].to_numpy()

        ps_louo_train_labels = post_and_sleeve_label[indices_ps_louo_train]
        ps_louo_full_train_labels = post_and_sleeve_label[indices_ps_louo_full_train]
        ps_louo_validation_labels = post_and_sleeve_label[indices_ps_louo_validation]
        ps_louo_test_labels = post_and_sleeve_label[indices_ps_louo_test]

        x_train = ps_louo_train
        x_validation = ps_louo_validation

        x_full_train = ps_louo_full_train
        x_test = ps_louo_test

        x_train_original = ps_louo_train_original
        x_test_original = ps_louo_test_original

        y_train = ps_louo_train_labels
        y_validation = ps_louo_validation_labels
        y_test = ps_louo_test_labels
        y_full_train = ps_louo_full_train_labels

    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test), (x_full_train, y_full_train, x_train_original, x_test_original, PS_Full_Train)