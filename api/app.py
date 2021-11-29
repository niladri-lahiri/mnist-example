from flask import Flask, request

import numpy as np
import sys
sys.path.insert(1, '/home/niladri/mlops/mnist-example/mnist-example/utils')
sys.path.insert(1, '/exp/mnist-example/utils')
import utils

app = Flask(__name__)
#svm_clf = utils.load('/home/niladri/mlops/mnist-example/models/SVM_0.001.pkl') # saved models in wsl
#decision_tree_clf = utils.load('/home/niladri/mlops/mnist-example/models/DecisionTree_18.0.pkl') #saved models in wsl

svm_clf = utils.load('/exp/models/SVM_0.001.pkl') # saved models in docker
decision_tree_clf = utils.load('/exp/models/DecisionTree_18.0.pkl') #saved models in docker


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/svm_predict", methods=['POST', 'GET'])
def svm_predict():
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1, -1)
    predicted = svm_clf.predict(image)
    return str(predicted[0])


@app.route("/decision_tree_predict", methods=['POST', 'GET'])
def decision_tree_predict():
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1, -1)
    predicted = decision_tree_clf.predict(image)
    return str(predicted[0])


app.run('0.0.0.0', debug = True, port = '5000')
