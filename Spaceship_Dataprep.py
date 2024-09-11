# -----------------------------------------------------------------------------
# We will add Data preparation to this file
# -----------------------------------------------------------------------------

from azureml.core import Run

# get the run context
new_run = Run.get_context()

# Get the workspace from the run
ws = new_run.experiment.workspace

# -----------------------------------------------------------------------------
# Data preparation stage
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#df = new_run.input_datasets['raw_data'].to_pandas_dataframe()

# Access the dataset (it's a FileDataset)
dataset = new_run.input_datasets['raw_data']

# Download the dataset (assuming it's a CSV)
data_paths = dataset.download(target_path='.')

# Load the first file into a pandas DataFrame (assuming it's CSV)
df = pd.read_csv(data_paths[0])


data_prep = df.drop(['PassengerId', 'Name'], axis = 1) 

null_values = df.isnull().sum()

# Handling the null values

data_prep['HomePlanet'] = data_prep['HomePlanet'].fillna(data_prep['HomePlanet'].mode()[0])
data_prep['CryoSleep'] = data_prep['CryoSleep'].fillna(data_prep['CryoSleep'].mode()[0])

# Split the cabin column
data_prep[['Deck', 'Seat_no', 'ship_side']] = data_prep['Cabin'].str.split('/', expand=True)

data_prep['Deck'] = data_prep['Deck'].fillna(data_prep['Deck'].mode()[0])
data_prep['Seat_no'] = data_prep['Seat_no'].fillna(data_prep['Seat_no'].mode()[0])
data_prep['ship_side'] = data_prep['ship_side'].fillna(data_prep['ship_side'].mode()[0])


data_prep['Seat_no'] = data_prep['Seat_no'].astype('float64')

data_prep['Destination'] = data_prep['Destination'].fillna(data_prep['Destination'].mode()[0])

data_prep['VIP'] = data_prep['VIP'].fillna(data_prep['VIP'].mode()[0])

# handling the numerical values in dataset

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

for column in num_cols:
    data_prep[column] = data_prep[column].fillna(data_prep[column].mean())

scaler = MinMaxScaler()

# Apply MinMaxScaler to all numeric columns
numeric_columns = data_prep.select_dtypes(include=['float64', 'int64']).columns
data_prep[numeric_columns] = scaler.fit_transform(data_prep[numeric_columns])

data_prep['Transported'] = data_prep['Transported'].replace({False : 0, True : 1})


#X = data_prep.drop('Transported', axis = 1)
#Y = data_prep['Transported']

#X = pd.get_dummies(X)

#All_independent_cols = X.columns


import joblib
#obj_col_file = './outputs/columns.pkl'
Nomalised_file = './outputs/Norm_Scaler.pkl'
#joblib.dump(value=All_independent_cols, filename=obj_col_file)
joblib.dump(value = scaler, filename=Nomalised_file)                                                        
                                                
# Get the arguments from pipeline job
from argparse import ArgumentParser as AP
parser = AP()
parser.add_argument('--datafolder', type = str)
args = parser.parse_args()

# Create the folder if it does not exist
import os
os.makedirs(args.datafolder, exist_ok=True)


# Create the path
# i Need to change the dataset name
path = os.path.join(args.datafolder, 'defaults_prep.csv')

data_prep.to_csv(path, index=False)


for cols in df.columns:
    new_run.log(cols, null_values[cols])
    
    
from azureml.core import Model
    
Normalised_scaler = Model.register(workspace = ws,
                       model_path = Nomalised_file,
                       model_name = "Spaceship_Scaler",
                       tags = {'Alogorithm': 'Random Forest', 'Task': 'Classification'},
                       description = 'We are doing min max scaler those data will be stored here'
    )
    
# Complete the run
new_run.complete()





