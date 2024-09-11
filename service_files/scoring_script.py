# -----------------------------------------------------------------------------
# Entry script for the Spaceship prediction service
# -----------------------------------------------------------------------------

import joblib
import json
import pandas as pd

# -----------------------------------------------------------------------------
# init function for model deployment
# -----------------------------------------------------------------------------

from azureml.core.model import Model
import joblib

def init():
    global ref_cols, predictor, scaler
    
    try:
        model_path = Model.get_model_path('Spaceship_RandomForest', version = 4)
        print(f"Model path: {model_path}")  # Debugging line
        predictor = joblib.load(model_path)
        print("Model loaded successfully.")
        
        ref_cols_path = Model.get_model_path('Spaceship_columns', version = 3)
        print(f"Reference columns path: {ref_cols_path}")  # Debugging line
        ref_cols = joblib.load(ref_cols_path)
        print("Reference columns loaded successfully.")
        
        scaler_path = Model.get_model_path('Spaceship_Scaler', version = 3)
        print(f"Scaler path: {scaler_path}")  # Debugging line
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
        
    except Exception as e:
        print(f"Error during initialization: {e}")


    
    
def run(raw_data):
    
    print(f"raw_data type: {type(raw_data)}")
    
    data_dict = json.load(raw_data)['data']
    test_data_prep = pd.DataFrame.from_dict(data_dict)
    
    test_data_prep = test_data_prep.drop(['PassengerId', 'Name'], axis = 1)
    
    test_data_prep['HomePlanet'] = test_data_prep['HomePlanet'].fillna(test_data_prep['HomePlanet'].mode()[0])
    test_data_prep['CryoSleep'] = test_data_prep['CryoSleep'].fillna(test_data_prep['CryoSleep'].mode()[0])
    test_data_prep['CryoSleep'] = test_data_prep['CryoSleep'].replace({False : 0, True : 1})
    
    test_data_prep[['Deck', 'Seat_no', 'ship_side']] = test_data_prep['Cabin'].str.split('/', expand=True)
    test_data_prep['Deck'] = test_data_prep['Deck'].fillna(test_data_prep['Deck'].mode()[0])
    test_data_prep['Seat_no'] = test_data_prep['Seat_no'].fillna(test_data_prep['Seat_no'].mode()[0])
    test_data_prep['ship_side'] = test_data_prep['ship_side'].fillna(test_data_prep['ship_side'].mode()[0])
    
    test_data_prep = test_data_prep.drop('Cabin', axis = 1)
    
    test_data_prep['Seat_no'] = test_data_prep['Seat_no'].astype('float64')
    
    test_data_prep['Destination'] = test_data_prep['Destination'].fillna(test_data_prep['Destination'].mode()[0])
    
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    for column in num_cols:
        test_data_prep[column] = test_data_prep[column].fillna(test_data_prep[column].mean())
        
    test_data_prep['VIP'] = test_data_prep['VIP'].fillna(test_data_prep['VIP'].mode()[0])
    test_data_prep['VIP'] = test_data_prep['VIP'].astype('object')
    
    test_data_prep = pd.get_dummies(test_data_prep)
    
    numeric_columns_test = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',
           'VRDeck', 'Seat_no']
    test_data_prep[numeric_columns_test] = scaler.transform(test_data_prep[numeric_columns_test])
    
    #X_columns = X.columns
    
    #test_data_prep = test_data_prep[ref_cols]
    
    deploy_cols = test_data_prep.columns
    
    missing_cols = ref_cols.difference(deploy_cols)
    
    for cols in missing_cols:
        test_data_prep[cols] = 0
    
    test_data_prep =  test_data_prep[ref_cols]
    
    predictions = predictor.predict(test_data_prep)
    classes = [False, True]
    
    predicted_classes = []
    
    for prediction in predictions:
        predicted_classes.append(classes[prediction])
        
    return(json.dumps(predicted_classes))



    
    



    


