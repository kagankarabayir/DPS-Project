# DPS-Project
Munich Traffic Accident Prediction Challenge

This repository contains the code and resources for the AI Engineering Challenge as part of the AI Track application at Digital Product School (DPS). The goal of this challenge is to develop a machine learning model to predict the number of traffic accidents based on historical data and evaluate its performance against actual data.

Project Overview

The project involves the following steps:
	1.	Data Preparation:
	•	The dataset includes traffic accident data in Munich up to 2021.
	•	To ensure consistency, all records after 2020 are excluded during data preprocessing.
	2.	Model Development:
	•	A regression model is trained using the preprocessed dataset to predict the number of traffic accidents.
	•	The model uses logarithmic scaling for training to ensure predictions are positive.
	3.	Error Evaluation:
	•	After training, the model is evaluated by comparing its predictions with actual data from the test set.
	•	Mean Squared Error (MSE) is used to measure performance.

Dataset

The Munich Open Data Portal provides the dataset used in this challenge and contains monthly traffic accident statistics. It includes various categories of accidents, and for this project, we focus specifically on alcohol-related accidents.

If the dataset link provided on the challenge page does not work, use this alternate link:
Monatszahlen Verkehrsunfälle Dataset - München Open Data Portal


Deployment

The application has been deployed on Google Cloud Platform. Visit the live application at:
Munich Traffic Accident Prediction App: https://dps-project-442316.uc.r.appspot.com/




