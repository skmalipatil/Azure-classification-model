# Azure-classification-model

This repository contains the code and scripts for a classification model deployed in Azure. The model pipeline handles data preparation, training, deployment, and scoring for predictions.

This project is integrated with Azure Machine Learning for scalable model training and deployment. The trained model is saved as a model.pkl file in Azure, and the scoring script is used for real-time predictions.

Project Structure
The project is organized into five key components:

Data Preparation (spaceship_Dataprep.py)

This file handles cleaning, preprocessing, and transforming the raw data for model training.
Tasks include handling missing values, encoding categorical variables, feature scaling, and feature engineering.

Data Training (Spaceship_training.py)

This script trains the classification model using the prepared data.
Different models can be trained, evaluated, and selected based on performance metrics such as accuracy, precision, recall, and F1 score.

Pipeline Creation and Model Storage (Spaceship_pipeline.py)

This file orchestrates the entire pipeline, incorporating both the data preparation and model training scripts.
After training, the model is saved as model.pkl in Azure for future predictions.
The pipeline is modular and designed for easy integration with Azure Machine Learning.
Deployment Configuration (Spaceship_deployment.py)

This script handles the configuration for deploying.
It sets up the Azure environment, specifies the compute target, and manages the deployment process.

Scoring Script (scoring_script.py)
This script is responsible for handling incoming data, applying any necessary transformations, and making predictions using the trained model.
The script outputs the prediction results for the given input data.

Setup and Usage
Prerequisites
Python 3.7+
Azure Machine Learning SDK
Scikit-learn, pandas, and other relevant libraries
