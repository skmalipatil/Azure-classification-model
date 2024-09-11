# --------------------------------------------------------------
# Model training step of the pipeline run
# --------------------------------------------------------------

# Import required classes from Azureml
from azureml.core import Run
import argparse

# get the context of experiment run
new_run = Run.get_context()

# Access the workspace
ws = new_run.experiment.workspace

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--datafolder", type=str)
args = parser.parse_args()

# -----------------------------------------------------
# Model training 
# -----------------------------------------------------
# Read the data from the previous step

import os
import pandas as pd

path = os.path.join(args.datafolder, 'defaults_prep.csv')
data_prep = pd.read_csv(path)

print('data reading from data prep is completed for training')

# Create X and Y - Similar to "edit columns" in Train Module
Y = data_prep[['Transported']]
X = data_prep.drop(['Transported'], axis=1)

X = X.drop('Cabin', axis = 1)

X = pd.get_dummies(X)

All_independent_cols = X.columns

# split the data into train and test data

from sklearn.model_selection import train_test_split 

X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = 0.3, random_state = 1234, stratify=Y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

clf = RandomForestClassifier(random_state=0)

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='f1')


print('Teraining the model')
grid_search.fit(X_train, Y_train)


print('Model training completed')

Y_predict = grid_search.predict(X_test)


Y_prob = grid_search.predict_proba(X_test)[:, 1]

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


cm = confusion_matrix(Y_test, Y_predict)
score = grid_search.score(X_test, Y_test)
f1 = f1_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict)
recall = recall_score(Y_test, Y_predict)


# Create the confusion matrix dictionary
cm_dict = {"schema_type": "confusion_matrix",
           "schema_version": "v1",
           "data": {"class_labels": ["N", "Y"],
                    "matrix": cm.tolist()}
           }


new_run.log("TotalObservations", len(data_prep))
new_run.log_confusion_matrix("ConfusionMatrix", cm_dict)
new_run.log("Score", score)
new_run.log("F1_Score", f1)
new_run.log("Precision", precision)
new_run.log("recall", recall)



# Upload the scored dataset
Y_prob_df    = pd.DataFrame(Y_prob, columns=["Scored Probabilities"]) 
Y_predict_df = pd.DataFrame(Y_predict, columns=["Scored Label"]) 

scored_dataset = pd.concat([X_test, Y_test, Y_predict_df, Y_prob_df],
                           axis=1)

scored_dataset.to_csv("./outputs/defaults_scored.csv",
                      index=False)


# -----------------------------------------------------------------------------
# adding the model explainability
# -----------------------------------------------------------------------------

from interpret.ext.blackbox import TabularExplainer
from azureml.interpret import ExplanationClient

# Extract the experiment name and run ID from the current run context
experiment_name = new_run.experiment.name
run_id = new_run.id


# Create an explainer for the trained model
explainer = TabularExplainer(grid_search.best_estimator_, X_train, features=X.columns)

# Global explanations (feature importance across the entire dataset)
global_explanation = explainer.explain_global(X_train)

# Local explanations (for individual predictions)
local_explanations = explainer.explain_local(X_test[:5])

# Log global explanations as a feature importance plot in Azure ML
#explanation_client = ExplanationClient(new_run)

explanation_client =ExplanationClient.from_run_id(
    workspace=ws,
    experiment_name=experiment_name,
    run_id=run_id
)


explanation_client.upload_model_explanation(global_explanation, comment="Global Explanation")

# -----------------------------------------------------
# Log local explanations for a few samples
# -----------------------------------------------------
#for i in range(5):
#    new_run.log("Local Explanation - Sample {}".format(i + 1), str(local_explanations.local_importance_values[i]))

# -----------------------------------------------------------------------------
# register the model
# -----------------------------------------------------------------------------

import joblib

# save the model to file

model_file = './outputs/Spaceship_RandomForest_model.pkl'
joblib.dump(value=grid_search, filename=model_file)


obj_col_file = './outputs/columns.pkl'
joblib.dump(value=All_independent_cols, filename=obj_col_file)

# -----------------------------------------------------------------------------
# Registering the model
# -----------------------------------------------------------------------------

from azureml.core import Model



model = Model.register(workspace = ws,
                       model_path = model_file,
                       model_name = "Spaceship_RandomForest",
                       tags = {'Alogorithm': 'Random Forest', 'Task': 'Classification'},
                       description = 'Random forest model trained on spaceship data'
    )



columns = Model.register(workspace = ws,
                       model_path = obj_col_file,
                       model_name = "Spaceship_columns",
                       tags = {'Alogorithm': 'Random Forest', 'Task': 'Classification'},
                       description = 'We are storing all the column names for later use'
    )



new_run.complete()
