import h5py
import pandas as pd

def load_UTKFace_data(DIR):
    with h5py.File(DIR, 'r') as h5:
        X_train = h5['X_train'][:]
        X_valid = h5['X_valid'][:]
        X_test = h5['X_test'][:]
        age_train = h5['age_train'][:]
        age_valid = h5['age_valid'][:]
        age_test = h5['age_test'][:]
        age_group_train = h5['age_group_train'][:]
        age_group_valid = h5['age_group_valid'][:]
        age_group_test = h5['age_group_test'][:]
        gender_train = h5['gender_train'][:]
        gender_valid = h5['gender_valid'][:]
        gender_test = h5['gender_test'][:]
        race_train = h5['race_train'][:]
        race_valid = h5['race_valid'][:]
        race_test = h5['race_test'][:]
        id_train = h5['id_train'][:]
        id_valid = h5['id_valid'][:]
        id_test = h5['id_test'][:]
        x_1_train = h5['x_1_train'][:]
        x_1_valid = h5['x_1_valid'][:]
        x_1_test = h5['x_1_test'][:]
        x_2_train = h5['x_2_train'][:]
        x_2_valid = h5['x_2_valid'][:]
        x_2_test = h5['x_2_test'][:]
        x_3_train = h5['x_3_train'][:]
        x_3_valid = h5['x_3_valid'][:]
        x_3_test = h5['x_3_test'][:]
        x_4_train = h5['x_4_train'][:]
        x_4_valid = h5['x_4_valid'][:]
        x_4_test = h5['x_4_test'][:]
        x_5_train = h5['x_5_train'][:]
        x_5_valid = h5['x_5_valid'][:]
        x_5_test = h5['x_5_test'][:]
        x_6_train = h5['x_6_train'][:]
        x_6_valid = h5['x_6_valid'][:]
        x_6_test = h5['x_6_test'][:]
        x_7_train = h5['x_7_train'][:]
        x_7_valid = h5['x_7_valid'][:]
        x_7_test = h5['x_7_test'][:]
        x_8_train = h5['x_8_train'][:]
        x_8_valid = h5['x_8_valid'][:]
        x_8_test = h5['x_8_test'][:]
        x_9_train = h5['x_9_train'][:]
        x_9_valid = h5['x_9_valid'][:]
        x_9_test = h5['x_9_test'][:]
        x_10_train = h5['x_10_train'][:]
        x_10_valid = h5['x_10_valid'][:]
        x_10_test = h5['x_10_test'][:]
        
    train = pd.DataFrame({'age': age_train, 'age_group': age_group_train, 'gender': gender_train, 'race': race_train, 
                      'id': id_train, 'x_1': x_1_train, 'x_2': x_2_train, 'x_3': x_3_train, 'x_4': x_4_train,
                      'x_5': x_5_train, 'x_6': x_6_train, 'x_7': x_7_train, 'x_8': x_8_train, 'x_9': x_9_train,
                      'x_10': x_10_train})
    valid = pd.DataFrame({'age': age_valid, 'age_group': age_group_valid, 'gender': gender_valid, 'race': race_valid, 
                      'id': id_valid, 'x_1': x_1_valid, 'x_2': x_2_valid, 'x_3': x_3_valid, 'x_4': x_4_valid,
                      'x_5': x_5_valid, 'x_6': x_6_valid, 'x_7': x_7_valid, 'x_8': x_8_valid, 'x_9': x_9_valid,
                      'x_10': x_10_valid})
    test = pd.DataFrame({'age': age_test, 'age_group': age_group_test, 'gender': gender_test, 'race': race_test, 
                      'id': id_test, 'x_1': x_1_test, 'x_2': x_2_test, 'x_3': x_3_test, 'x_4': x_4_test,
                      'x_5': x_5_test, 'x_6': x_6_test, 'x_7': x_7_test, 'x_8': x_8_test, 'x_9': x_9_test,
                      'x_10': x_10_test})
    return X_train, train, X_valid, valid, X_test, test
     