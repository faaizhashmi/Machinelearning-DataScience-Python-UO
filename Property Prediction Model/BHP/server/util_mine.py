import json
import pickle
import numpy as np

__locations=None
__data_columns=None
__model=None

def load_save_artifacts():

    print("loading saved artifacts.. start")
    global __data_columns
    global __locations
    with open("./artifacts/columns.json", 'r') as f:
        __data_columns=json.load(f)['data_columns']
        __locations=__data_columns[3:]

    global __model
    if __model is None:
        with open("./artifacts/banglore_home_prices_model.pickle", 'rb') as f:
            __model = pickle.load(f)

    print("loading saved artifacts...done")


def get_estimated_price(location, total_sqft, bhk, bath):
    # load_save_artifacts()

    try:
        loc_index= __data_columns.index(location.lower())
    except:
        loc_index=-1


    x = np.zeros(len(__data_columns))
    print(x)
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

# 	x) make a function to give access to locations


def get_location_names():
    # global __locations
    load_save_artifacts()
    return __locations

# 	1) load saved artifacts datacolumns, locations and model


# 	2) will have price prediction function
if __name__=='__main__':
    load_save_artifacts()
    # print(__locations)
    print(__model)
    print(get_location_names())
    print(get_estimated_price('1st block jayanagar', 1000, 2, 3))
    # print(get_estimated_price('2nd stage nagarbhavi', 1000, 2, 3))
    # print(get_estimated_price('5th phase jp nagar', 1000, 2, 3))
    # print(get_estimated_price('konanakunte', 1000, 2, 3))