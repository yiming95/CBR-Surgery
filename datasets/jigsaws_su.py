"""
JIGSAWS Suturing dataset
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
    Load and return the JIGSAWS Suturing dataset.
    ## fold_number: "LOSO1"
    ==============             ==============
    Training Samples total                 31
    Testing Samples total                   8 
    Number of time steps                  9012
    Dimensionality                         76
    Number of targets                       3
    ==============             ==============

    # Returns
        # Numpy arrays: (x_train, y_train), (x_validation, y_validation), (x_test, y_test)
        (x_train, y_train), (x_validation, y_validation), (x_test, y_test), (x_full_train, x_train_original, x_test_original)
    """

    module_path = os.getcwd()

    ## please visit the offical dataset website of jigsaws to download, and organise it into dataset-self-proclaimed.pkl
    df_dataset_self = pd.read_pickle(module_path + '/datasets/jigsaws-kinematic/dataset-self-proclaimed.pkl')

    ### Suturing
    df_dataset_self_suturing = df_dataset_self.iloc[:39]

    kinemtiac_raw_data_self_suturing = df_dataset_self_suturing['kinematic_data'].to_numpy()
    padding_data_suturing = padding_wrap(kinemtiac_raw_data_self_suturing)

    reshaped_data_su_padding = padding_data_suturing.transpose(0, 2, 1)
    padding_data_suturing_mtype = convert(reshaped_data_su_padding, from_type="numpy3D", to_type="pd-multiindex")
    padding_data_suturing_nested_univ = convert(padding_data_suturing_mtype, from_type="pd-multiindex", to_type="nested_univ")      
    
    suturing_self_claimed_trail_name = df_dataset_self_suturing['Trail'].to_numpy()


    if fold_number[-1].isdigit(): ## LOSO1
        print("************ LOSO Cross Validation ************")
        loso_suffix = "00" + fold_number[-1]
        SU_LOSO_Test, SU_Full_Train = get_LOSO_sets(suturing_self_claimed_trail_name, loso_suffix)
        SU_LOSO_Validation, SU_LOSO_Train = select_validation_set(SU_Full_Train)

        indices_su_loso_test = [np.where(suturing_self_claimed_trail_name == a)[0][0] for a in SU_LOSO_Test]
        indices_su_loso_full_train = [np.where(suturing_self_claimed_trail_name == a)[0][0] for a in SU_Full_Train]  
        
        print("SU LOSO Full Train Trial Names: ", SU_Full_Train)
        print("Indices SU Full Train: ", indices_su_loso_full_train)

        indices_su_loso_train = [np.where(suturing_self_claimed_trail_name == a)[0][0] for a in SU_LOSO_Train]
        indices_su_loso_validation = [np.where(suturing_self_claimed_trail_name == a)[0][0] for a in SU_LOSO_Validation]
        
        su_loso_train = padding_data_suturing_nested_univ.iloc[indices_su_loso_train]
        su_loso_validation = padding_data_suturing_nested_univ.iloc[indices_su_loso_validation]

        su_loso_full_train = padding_data_suturing_nested_univ.iloc[indices_su_loso_full_train]
        su_loso_test = padding_data_suturing_nested_univ.iloc[indices_su_loso_test]

        ## in original shape (32, 9012, 76)
        su_loso_train_original = padding_data_suturing[indices_su_loso_full_train, :, :]
        su_loso_test_original = padding_data_suturing[indices_su_loso_test, :, :]

        ## label
        suturing_self_claimed_label = df_dataset_self_suturing['skill_level'].to_numpy()

        su_loso_train_labels = suturing_self_claimed_label[indices_su_loso_train]
        su_loso_full_train_labels = suturing_self_claimed_label[indices_su_loso_full_train]

        su_loso_validation_labels = suturing_self_claimed_label[indices_su_loso_validation]
        su_loso_test_labels = suturing_self_claimed_label[indices_su_loso_test]

        x_train = su_loso_train
        x_validation = su_loso_validation

        x_full_train = su_loso_full_train
        x_test = su_loso_test

        x_train_original = su_loso_train_original
        x_test_original = su_loso_test_original

        y_train = su_loso_train_labels
        y_validation = su_loso_validation_labels
        y_test = su_loso_test_labels

        y_full_train = su_loso_full_train_labels

    elif fold_number[-1].isalpha(): ## LOUOB
        print("************ LOUO Cross Validation ************")
        louo_suffix = fold_number[-1]
        SU_LOUO_Test, SU_Full_Train = get_LOUO_sets(suturing_self_claimed_trail_name, louo_suffix)
        
        SU_LOUO_Validation, SU_LOUO_Train = select_validation_set(SU_Full_Train)

        indices_su_louo_test = [np.where(suturing_self_claimed_trail_name == a)[0][0] for a in SU_LOUO_Test]
        indices_su_louo_full_train = [np.where(suturing_self_claimed_trail_name == a)[0][0] for a in SU_Full_Train]  
        
        print("SU LOUO Full Train Trial Names: ", SU_Full_Train)
        print("Indices SU Full Train: ", indices_su_louo_full_train)

        indices_su_louo_train = [np.where(suturing_self_claimed_trail_name == a)[0][0] for a in SU_LOUO_Train]
        indices_su_louo_validation = [np.where(suturing_self_claimed_trail_name == a)[0][0] for a in SU_LOUO_Validation]
        
        su_louo_train = padding_data_suturing_nested_univ.iloc[indices_su_louo_train]
        su_louo_validation = padding_data_suturing_nested_univ.iloc[indices_su_louo_validation]

        su_louo_full_train = padding_data_suturing_nested_univ.iloc[indices_su_louo_full_train]
        su_louo_test = padding_data_suturing_nested_univ.iloc[indices_su_louo_test]

        ## in original shape (32, 9012, 76)
        su_louo_train_original = padding_data_suturing[indices_su_louo_full_train, :, :]
        su_louo_test_original = padding_data_suturing[indices_su_louo_test, :, :]

        ## label
        suturing_self_claimed_label = df_dataset_self_suturing['skill_level'].to_numpy()

        su_louo_train_labels = suturing_self_claimed_label[indices_su_louo_train]
        su_louo_full_train_labels = suturing_self_claimed_label[indices_su_louo_full_train]

        su_louo_validation_labels = suturing_self_claimed_label[indices_su_louo_validation]
        su_louo_test_labels = suturing_self_claimed_label[indices_su_louo_test]

        x_train = su_louo_train
        x_validation = su_louo_validation

        x_full_train = su_louo_full_train
        x_test = su_louo_test

        x_train_original = su_louo_train_original
        x_test_original = su_louo_test_original

        y_train = su_louo_train_labels
        y_validation = su_louo_validation_labels
        y_test = su_louo_test_labels

        y_full_train = su_louo_full_train_labels

    
    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test), (x_full_train, y_full_train, x_train_original, x_test_original, SU_Full_Train)