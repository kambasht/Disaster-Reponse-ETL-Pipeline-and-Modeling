# Disaster-Reponse-Pipeline-and-Modeling
Data Pipeline and modeling in Python to help classify disaster response messages

# Table of Contents
1. Project Motivation
2. Folder setup
3. File Descriptions
4. Results
5. Licensing, Authors, and Acknowledgements

## Project Motivation

Natural disasters have affected everyone across the world. In the times of distress and chaos, victims send out millions of messages hoping to get in touch with the nearest disaster recovery centre. This project utilizes a data set which contains real messages that were sent during disaster events. We are utilizing a machine learning pipeline to categorize these events so that we can categorize and send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Folder setup

The relevant files have been attached in the github code. Make sure you create the following folder structure on your system to successfully run the ETL, modeling and the web app.

Folder Structure:

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 


## File Descriptions

ETL Pipeline
process_data.py

    Loads the messages and categories datasets
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database

ML Pipeline
train_classifier.py (a machine learning pipeline) that:

    Loads data from the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file

Flask Web App
run.py

The file utilizes Flask flask web app to display an interactive website for an emergency worker.

## Results


## Licensing, Authors, Acknowledgements
Credits to Figure Eight (https://www.figure-eight.com/dataset/combined-disaster-response-data/) for the dataset and [Udacity](https://www.udacity.com/) for the skeleton structure for the code.
