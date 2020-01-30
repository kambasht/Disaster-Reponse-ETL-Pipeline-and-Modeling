import sys,time

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import re,pickle
from sqlalchemy import create_engine
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    INPUT
    database_filepath- file path where the database is saved
    
    OUTPUT
    X- the message column of dataframe
    Y- the dataframe without the columns 'message', 'genre', 'id', 'original'
    category names- list of column names after dropping the columns 'message', 'genre', 'id', 'original'
    
    This function :
    1. reads the database into a dataframe
    2. drops columns not needed
    2. divides the dataset into train and test data
    3. returns the target and independent variables X and Y for model training
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterReponseMessages', 'sqlite:///'+database_filepath)
    
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)
    category_names= Y.columns
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size = 0.2, random_state = 45)
    
    return X,Y,category_names


def tokenize(text):
    '''
    INPUT
    text- the text that needs to be tokenized
    
    OUTPUT
    tokens - a list of tokenized words after cleaning up the input text
    
    This function :
    1. converts text to all lower case and removes punctuations
    2. tokenize entire text into words
    2. lemmatize each word using WordNetLemmatizer
    3. remove all stop words from text as per english corpus of stop words
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens


def build_model():
    '''
    INPUT
    None
    
    OUTPUT
    pipeline- a pipeline built to tokenize, transform and classify text data
    
    This function :
    1. builds a pipeline of countvectorizer, tfidf transformer, and
    2. a random forest multi output classifier
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    # below parameters have been selected keeping the model training time in mind
    # these can be modified to improve accuracy and precision
   
    parameters = {
              #'tfidf__use_idf': (True, False),
              'clf__estimator__n_estimators': [1, 2],       #[50, 100]
              'clf__estimator__min_samples_split': [2, 3]   #[2, 4]
              }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model
    X_test
    Y_test
    category_names
    
    OUTPUT
    
    
    This function :
    1. reads the database into a dataframe
    2. drops columns not needed
    2. divides the dataset into train and test data
    3. returns the target and independent variables X and Y for model training
    '''
    y_pred = model.predict(X_test)
    for n,col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], y_pred[:, n]))


def save_model(model, model_filepath):
    '''
    INPUT
    model
    model_filepath
    
    OUTPUT
    
    
    This function :
    1. reads the database into a dataframe
    2. drops columns not needed
    2. divides the dataset into train and test data
    3. returns the target and independent variables X and Y for model training
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        start_time = time.time()
        model.fit(X_train, Y_train)
        elapsed_time = round((time.time() - start_time)/60 , 2)
        print('Training time...{} minutes'.format(elapsed_time))
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()