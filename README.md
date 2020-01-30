Data Pipeline and modeling in Python to help classify disaster response messages

# Table of Contents
1. Project Motivation
2. Folder setup
3. File Descriptions
4. Running the code
5. Results
6. Licensing, Authors, and Acknowledgements

## Project Motivation

Natural disasters have affected everyone across the world. In the times of distress and chaos, victims send out millions of messages hoping to get in touch with the nearest disaster recovery centre. This project utilizes a data set which contains real messages that were sent during disaster events. We are utilizing a machine learning pipeline to categorize these events into one of the 36 categories of food, shelter, electricity, storm etc. so that these messages can be then sent to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Folder setup

The relevant files have been attached in the github code. Make sure you create the following folder structure on your system to successfully run the ETL, modeling and the web app.

Folder Structure:

- app
    - template
       - master.html  # main page of web app
       - go.html  # classification result page of web app
    - run.py  # Flask file that runs app

- data
    - disaster_categories.csv  # data to process 
    - disaster_messages.csv  # data to process
    - process_data.py
    - InsertDatabaseName.db   # database to save clean data to

- models
    - train_classifier.py # ml script to train classifier
    - classifier.pkl  # saved model 


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

## Running the code

Installations required before running the code:
- pip install sqlalchemy
- pip install plotly

Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    - change the directory to app
        'cd app'
    - run the script
        'python run.py'

3. Go to 'localhost:3001' in one of your browser window

## Results
While some of the user input messages are being classified correctly into one of the 36 categories in the list, there is a scope of improvement to enhance the model through a better gridsearch process. The current process has been optimized to consume less time in model build and fitting data.

## Licensing, Authors, Acknowledgements
Credits to [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/) for the dataset and [Udacity](https://www.udacity.com/) for the skeleton structure for the code.
