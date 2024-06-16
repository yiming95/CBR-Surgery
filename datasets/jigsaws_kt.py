"""
JIGSAWS Knot tying dataset
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
    Load and return the JIGSAWS Knot tying dataset.
    ## fold_number: "LOSO1"
    ==============             ==============
    Total Trials                           36
    Training Samples total                 29
    Testing Samples total                   7 
    Number of time steps                  3853
    Dimensionality                         76
    Number of targets                       3
    ==============             ==============
    """

    module_path = os.getcwd()


    ## please visit the offical dataset website of jigsaws to download, and organise it into dataset-self-proclaimed.pkl
    df_dataset_self = pd.read_pickle(module_path + '/datasets/jigsaws-kinematic/dataset-self-proclaimed.pkl')

    ### Knot tying
    df_dataset_self_knot_tying = df_dataset_self.iloc[39:75]

    kinemtiac_raw_data_self_knot_tying = df_dataset_self_knot_tying['kinematic_data'].to_numpy()
    padding_data_knot_tying = padding_wrap(kinemtiac_raw_data_self_knot_tying)

    reshaped_data_kt_padding = padding_data_knot_tying.transpose(0, 2, 1)
    padding_data_knot_tying_mtype = convert(reshaped_data_kt_padding, from_type="numpy3D", to_type="pd-multiindex")
    padding_data_knot_tying_nested_univ = convert(padding_data_knot_tying_mtype, from_type="pd-multiindex", to_type="nested_univ")      
    
    knot_tying_self_claimed_trail_name = df_dataset_self_knot_tying['Trail'].to_numpy()

    if fold_number[-1].isdigit(): ## LOSO1

        print("************ LOSO Cross Validation ************")

        loso_suffix = "00" + fold_number[-1]
        KT_LOSO_Test, KT_Full_Train = get_LOSO_sets(knot_tying_self_claimed_trail_name, loso_suffix)
        KT_LOSO_Validation, KT_LOSO_Train = select_validation_set(KT_Full_Train)

        print("KT Full Train Trial Names: ", KT_Full_Train)

        indices_kt_loso_test = [np.where(knot_tying_self_claimed_trail_name == a)[0][0] for a in KT_LOSO_Test]
        indices_kt_loso_full_train = [np.where(knot_tying_self_claimed_trail_name == a)[0][0] for a in KT_Full_Train]  

        indices_kt_loso_train = [np.where(knot_tying_self_claimed_trail_name == a)[0][0] for a in KT_LOSO_Train]
        indices_kt_loso_validation = [np.where(knot_tying_self_claimed_trail_name == a)[0][0] for a in KT_LOSO_Validation]
        
        kt_loso_train = padding_data_knot_tying_nested_univ.iloc[indices_kt_loso_train]
        kt_loso_validation = padding_data_knot_tying_nested_univ.iloc[indices_kt_loso_validation]

        kt_loso_full_train = padding_data_knot_tying_nested_univ.iloc[indices_kt_loso_full_train]
        kt_loso_test = padding_data_knot_tying_nested_univ.iloc[indices_kt_loso_test]

        ## in original shape
        kt_loso_train_original = padding_data_knot_tying[indices_kt_loso_full_train, :, :]
        kt_loso_test_original = padding_data_knot_tying[indices_kt_loso_test, :, :]

        ## label
        knot_tying_self_claimed_label = df_dataset_self_knot_tying['skill_level'].to_numpy()

        kt_loso_train_labels = knot_tying_self_claimed_label[indices_kt_loso_train]
        kt_loso_full_train_labels = knot_tying_self_claimed_label[indices_kt_loso_full_train]
        kt_loso_validation_labels = knot_tying_self_claimed_label[indices_kt_loso_validation]
        kt_loso_test_labels = knot_tying_self_claimed_label[indices_kt_loso_test]

        x_train = kt_loso_train
        x_validation = kt_loso_validation

        x_full_train = kt_loso_full_train
        x_test = kt_loso_test

        x_train_original = kt_loso_train_original
        x_test_original = kt_loso_test_original

        y_train = kt_loso_train_labels
        y_validation = kt_loso_validation_labels
        y_test = kt_loso_test_labels
        y_full_train = kt_loso_full_train_labels

    elif fold_number[-1].isalpha(): ## LOUOB

        print("************ LOUO Cross Validation ************")

        louo_suffix = fold_number[-1]
        KT_LOUO_Test, KT_Full_Train = get_LOUO_sets(knot_tying_self_claimed_trail_name, louo_suffix)
        
        KT_LOUO_Validation, KT_LOUO_Train = select_validation_set(KT_Full_Train)

        indices_kt_louo_test = [np.where(knot_tying_self_claimed_trail_name == a)[0][0] for a in KT_LOUO_Test]
        indices_kt_louo_full_train = [np.where(knot_tying_self_claimed_trail_name == a)[0][0] for a in KT_Full_Train]  
        
        print("KT LOUO Full Train Trial Names: ", KT_Full_Train)
        print("Indices KT Full Train: ", indices_kt_louo_full_train)

        indices_kt_louo_train = [np.where(knot_tying_self_claimed_trail_name == a)[0][0] for a in KT_LOUO_Train]
        indices_kt_louo_validation = [np.where(knot_tying_self_claimed_trail_name == a)[0][0] for a in KT_LOUO_Validation]
        
        kt_louo_train = padding_data_knot_tying_nested_univ.iloc[indices_kt_louo_train]
        kt_louo_validation = padding_data_knot_tying_nested_univ.iloc[indices_kt_louo_validation]

        kt_louo_full_train = padding_data_knot_tying_nested_univ.iloc[indices_kt_louo_full_train]
        kt_louo_test = padding_data_knot_tying_nested_univ.iloc[indices_kt_louo_test]

        ## in original shape (32, 9012, 76)
        kt_louo_train_original = padding_data_knot_tying[indices_kt_louo_full_train, :, :]
        kt_louo_test_original = padding_data_knot_tying[indices_kt_louo_test, :, :]

        ## label
        knot_tying_self_claimed_label = df_dataset_self_knot_tying['skill_level'].to_numpy()

        kt_louo_train_labels = knot_tying_self_claimed_label[indices_kt_louo_train]
        kt_louo_full_train_labels = knot_tying_self_claimed_label[indices_kt_louo_full_train]

        kt_louo_validation_labels = knot_tying_self_claimed_label[indices_kt_louo_validation]
        kt_louo_test_labels = knot_tying_self_claimed_label[indices_kt_louo_test]

        x_train = kt_louo_train
        x_validation = kt_louo_validation

        x_full_train = kt_louo_full_train
        x_test = kt_louo_test

        x_train_original = kt_louo_train_original
        x_test_original = kt_louo_test_original

        y_train = kt_louo_train_labels
        y_validation = kt_louo_validation_labels
        y_test = kt_louo_test_labels

        y_full_train = kt_louo_full_train_labels

    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test), (x_full_train, y_full_train, x_train_original, x_test_original, KT_Full_Train)