# Serve model as a flask application

import gzip, pickle
import numpy as np
from flask import Flask, request
from sklearn import datasets

model = None
app = Flask(__name__)

#Load libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import joblib
import os

def train_model():
    global model

    iris = datasets.load_iris()
    features = iris.data
    target = iris.target

    #Create decision tree classifer object
    classifer = RandomForestClassifier()

    #Train model
    model = classifer.fit(features, target)

    #Save model as pickle file
    path = os.getcwd()
    # joblib.dump(model, os.path.join(path,"iris_trained_model.pkl"))
    with gzip.open(os.path.join(path,"iris_trained_model.pklz"), 'wb') as ofp:
        pickle.dump(model, ofp)


def load_model():
    global model
    # model variable refers to the global variable
    # with open(os.path.join(os.getcwd(),"iris_trained_model.pkl"), 'rb') as f:
    #     model = pickle.load(f)
    with gzip.open(os.path.join(os.getcwd(),"iris_trained_model.pklz"), 'rb') as ifp:
        model = pickle.load(ifp)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict')
def get_prediction():
    # # Works only for a single sample
    # if request.method == 'POST':
    #     data = request.get_json()  # Get data posted as a json
    #     data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
    #     prediction = model.predict(data)  # runs globally loaded model on the data
    # return str(prediction[0])
    total = ''
    data = [5.9,3.0,5.1,1.8]
    data = np.array(data)[np.newaxis, :]
    prediction = model.predict(data)
    # print(prediction,' \n',prediction[0])
    total += str(prediction[0])
    for i in range(10):
        data = np.random.randn(4)
        data = np.array(data)[np.newaxis, :]
        print(data)
        prediction = model.predict(data)
        total += ' ' + str(prediction[0])
    return total
 
if __name__ == '__main__':
    train_model()
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=5000)