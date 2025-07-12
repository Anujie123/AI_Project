import pandas as pd
import os

def preprocess_data():
    # Create data directory if not exists
    os.makedirs('data', exist_ok=True)
    
    # Load and process data
    red = pd.read_csv('data/winequality-red.csv', sep=';')
    white = pd.read_csv('data/winequality-white.csv', sep=';')
    
    red['type'] = 1
    white['type'] = 0
    data = pd.concat([red, white])
    
    data['sulfur_ratio'] = data['free sulfur dioxide'] / data['total sulfur dioxide']
    data.to_csv('data/processed_data.csv', index=False)

if __name__ == '__main__':
    preprocess_data()