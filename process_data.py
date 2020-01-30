import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - The filepath where the messages file is located
    categories_filepath - The filepath where the categories file is located
    
    OUTPUT
    df - a merged dataframe that has the messages and categories merged together
    
    This function :
    1. loads the data from two different csv files that contain categories and messages
    2. returns a dataframe that has the categories and messages combined
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on="id")
    return df

def clean_data(df):
    '''
    INPUT
    df - dataframe which has all category names and other columns
    
    OUTPUT
    df - a cleaned dataframe that has no duplicates and cleaned up column names with int values
    
    This function :
    1. cleans category columns
    2. returns a dataframe that has cleaned up category columns as 0 and 1 value
    '''
    categories = df['categories'].str.split(";",expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    df.drop(['categories'],axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    return df
    
def save_data(df, database_filename):
    '''
    INPUT
    df - dataframe
    database_filename- name of the database file
    
    OUTPUT
    None
    
    This function :
    1. creates a sqlalchemy database engine
    2. saves the cleaned dataframe as a database
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterReponseMessages', engine,if_exists='replace', index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()