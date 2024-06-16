"""
JIGSAWS Needle passing dataset
"""
import os 
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
import pickle
import pandas as pd
import random
from sktime.datatypes import convert

def get_LOSO_sets(data_list, loso_suffix):
    test_set = [s for s in data_list if s.endswith(loso_suffix)]
    train_set = [s for s in data_list if not s.endswith(loso_suffix)]
    return test_set, train_set

def get_LOUO_sets(data_list, louo_suffix):
    test_set = [s for s in data_list if louo_suffix in s]
    train_set = [s for s in data_list if louo_suffix not in s]
    return test_set, train_set

def select_validation_set(train_set, num_trials=6):
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
    Load and return the JIGSAWS Needle passing dataset.
    ## fold_number: "LOSO1"
    ==============             ==============
    Total Trials                           28
    Training Samples total                 23
    Testing Samples total                   5 
    Number of time steps                  4761
    Dimensionality                         76
    Number of targets                       3
    ==============             ==============

    """

    module_path = os.getcwd()

    ## please visit the offical dataset website of jigsaws to download, and organise it into dataset-self-proclaimed.pkl
    df_dataset_self = pd.read_pickle(module_path + '/datasets/jigsaws-kinematic/dataset-self-proclaimed.pkl')

    ### Needle passing
    df_dataset_self_needle_passing = df_dataset_self.iloc[75:]

    kinemtiac_raw_data_self_needle_passing = df_dataset_self_needle_passing['kinematic_data'].to_numpy()
    padding_data_needle_passing = padding_wrap(kinemtiac_raw_data_self_needle_passing)

    reshaped_data_np_padding = padding_data_needle_passing.transpose(0, 2, 1)
    padding_data_needle_passing_mtype = convert(reshaped_data_np_padding, from_type="numpy3D", to_type="pd-multiindex")
    padding_data_needle_passing_nested_univ = convert(padding_data_needle_passing_mtype, from_type="pd-multiindex", to_type="nested_univ")      
    
    needle_passing_self_claimed_trail_name = df_dataset_self_needle_passing['Trail'].to_numpy()


    if fold_number[-1].isdigit(): ## LOSO1

        print("************ LOSO Cross Validation ************")

        loso_suffix = "00" + fold_number[-1]
        NP_LOSO_Test, NP_Full_Train = get_LOSO_sets(needle_passing_self_claimed_trail_name, loso_suffix)
        NP_LOSO_Validation, NP_LOSO_Train = select_validation_set(NP_Full_Train)

        print("Indices NP Full Train: ", NP_Full_Train)

        indices_np_loso_test = [np.where(needle_passing_self_claimed_trail_name == a)[0][0] for a in NP_LOSO_Test]
        indices_np_loso_full_train = [np.where(needle_passing_self_claimed_trail_name == a)[0][0] for a in NP_Full_Train]  

        indices_np_loso_train = [np.where(needle_passing_self_claimed_trail_name == a)[0][0] for a in NP_LOSO_Train]
        indices_np_loso_validation = [np.where(needle_passing_self_claimed_trail_name == a)[0][0] for a in NP_LOSO_Validation]
        
        np_loso_train = padding_data_needle_passing_nested_univ.iloc[indices_np_loso_train]
        np_loso_validation = padding_data_needle_passing_nested_univ.iloc[indices_np_loso_validation]

        np_loso_full_train = padding_data_needle_passing_nested_univ.iloc[indices_np_loso_full_train]
        np_loso_test = padding_data_needle_passing_nested_univ.iloc[indices_np_loso_test]

        ## in original shape
        np_loso_train_original = padding_data_needle_passing[indices_np_loso_full_train, :, :]
        np_loso_test_original = padding_data_needle_passing[indices_np_loso_test, :, :]

        ## label
        needle_passing_self_claimed_label = df_dataset_self_needle_passing['skill_level'].to_numpy()

        np_loso_train_labels = needle_passing_self_claimed_label[indices_np_loso_train]
        np_loso_full_train_labels = needle_passing_self_claimed_label[indices_np_loso_full_train]
        np_loso_validation_labels = needle_passing_self_claimed_label[indices_np_loso_validation]
        np_loso_test_labels = needle_passing_self_claimed_label[indices_np_loso_test]

        x_train = np_loso_train
        x_validation = np_loso_validation

        x_full_train = np_loso_full_train
        x_test = np_loso_test

        x_train_original = np_loso_train_original
        x_test_original = np_loso_test_original

        y_train = np_loso_train_labels
        y_validation = np_loso_validation_labels
        y_test = np_loso_test_labels
        y_full_train = np_loso_full_train_labels    

    elif fold_number[-1].isalpha(): ## LOUOB

        print("************ LOUO Cross Validation ************")

        louo_suffix = fold_number[-1]
        NP_LOUO_Test, NP_Full_Train = get_LOUO_sets(needle_passing_self_claimed_trail_name, louo_suffix)
        
        NP_LOUO_Validation, NP_LOUO_Train = select_validation_set(NP_Full_Train)

        indices_np_louo_test = [np.where(needle_passing_self_claimed_trail_name == a)[0][0] for a in NP_LOUO_Test]
        indices_np_louo_full_train = [np.where(needle_passing_self_claimed_trail_name == a)[0][0] for a in NP_Full_Train]  
        
        print("NP LOUO Full Train Trial Names: ", NP_Full_Train)
        print("Indices NP Full Train: ", indices_np_louo_full_train)

        indices_np_louo_train = [np.where(needle_passing_self_claimed_trail_name == a)[0][0] for a in NP_LOUO_Train]
        indices_np_louo_validation = [np.where(needle_passing_self_claimed_trail_name == a)[0][0] for a in NP_LOUO_Validation]
        
        np_louo_train = padding_data_needle_passing_nested_univ.iloc[indices_np_louo_train]
        np_louo_validation = padding_data_needle_passing_nested_univ.iloc[indices_np_louo_validation]

        np_louo_full_train = padding_data_needle_passing_nested_univ.iloc[indices_np_louo_full_train]
        np_louo_test = padding_data_needle_passing_nested_univ.iloc[indices_np_louo_test]

        ## in original shape (32, 9012, 76)
        np_louo_train_original = padding_data_needle_passing[indices_np_louo_full_train, :, :]
        np_louo_test_original = padding_data_needle_passing[indices_np_louo_test, :, :]

        ## label
        needle_passing_self_claimed_label = df_dataset_self_needle_passing['skill_level'].to_numpy()

        np_louo_train_labels = needle_passing_self_claimed_label[indices_np_louo_train]
        np_louo_full_train_labels = needle_passing_self_claimed_label[indices_np_louo_full_train]

        np_louo_validation_labels = needle_passing_self_claimed_label[indices_np_louo_validation]
        np_louo_test_labels = needle_passing_self_claimed_label[indices_np_louo_test]

        x_train = np_louo_train
        x_validation = np_louo_validation

        x_full_train = np_louo_full_train
        x_test = np_louo_test

        x_train_original = np_louo_train_original
        x_test_original = np_louo_test_original

        y_train = np_louo_train_labels
        y_validation = np_louo_validation_labels
        y_test = np_louo_test_labels

        y_full_train = np_louo_full_train_labels

    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test), (x_full_train, y_full_train, x_train_original, x_test_original, NP_Full_Train)