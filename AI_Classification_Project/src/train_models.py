import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_models():
    data = pd.read_csv('data/processed_data.csv')
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    # SVM
    svm = SVC()
    svm.fit(X, y)
    joblib.dump(svm, 'models/svm_model.pkl')
    
    # Naïve Bayes
    nb = GaussianNB()
    nb.fit(X, y)
    joblib.dump(nb, 'models/nb_model.pkl')
    
    # DNN
    y_encoded = pd.factorize(y)[0]
    dnn = Sequential([
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(set(y_encoded)), activation='softmax')
    ])
    dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    dnn.fit(X, y_encoded, epochs=10)
    dnn.save('models/dnn_model.h5')

if __name__ == '__main__':
    train_models()