import pickle, os

import pandas as pd
from skimage.transform import rescale
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import svm


def flatten_images():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    return data, digits


def preprocess(images, rescale_factor):

    resized_images = []
    for d in images:
        resized_images.append(rescale(d, rescale_factor, anti_aliasing = False))
    return resized_images


def create_splits(data, digits, test_size, validation_size_from_test_size):

    # Split data into 70% train and 15% validation and 15% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=test_size, shuffle=False)

    X_test, X_validation, y_test, y_validation = train_test_split(
        X_test, y_test, test_size=validation_size_from_test_size, shuffle=False)
    
    return X_train, X_test, X_validation, y_train, y_test, y_validation



def train_and_save_model(gamma_values, X_train, y_train, X_validation, y_validation):

    accuracy_validation, f1_validation, model_location, non_skipped_gamma_values = [], [], [], []

    for gamma in gamma_values: 

        # Create a classifier: a support vector classifier
        clf = svm.SVC(gamma=gamma)

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted_val = clf.predict(X_validation)

        val_accuracy_score = accuracy_score(y_validation, predicted_val)
        val_f1_score = f1_score(y_validation, predicted_val, average='weighted')

        if val_accuracy_score < 0.11:
            print("Skipping for gamma:", gamma)
            continue

        #Save the model to disk
        saved_model = pickle.dumps(clf)
        model_path = '/home/niladri/mlops/mnist-example/models'
        path = model_path + '/' + str(gamma) + '.pkl' 
        with open(path, 'wb') as f:
            pickle.dump(clf, f)

        
        accuracy_validation.append(val_accuracy_score)
        f1_validation.append(val_f1_score)
        model_location.append(path)
        non_skipped_gamma_values.append(gamma)

    df = pd.DataFrame(data = {'Gamma Value': non_skipped_gamma_values ,'Accuracy of Validation Data': accuracy_validation, 'f1 score of Validation Data': f1_validation, 'Model Location': model_location})

    return df


def get_best_model_path(df):

    return df.iloc[df['Accuracy of Validation Data'].argmax()]['Model Location']


def get_best_model_metrics(df):

    return df.iloc[df['Accuracy of Validation Data'].argmax()]['Accuracy of Validation Data'], df.iloc[df['Accuracy of Validation Data'].argmax()]['f1 score of Validation Data']


def model_test(model_path, X, y):

    clf = pickle.load(open(model_path, 'rb'))

    predicted_test = clf.predict(X)

    accuracy_test = accuracy_score(y, predicted_test)
    f1_score_test = f1_score(y, predicted_test, average='weighted')
    
    return accuracy_test, f1_score_test


def print_data_frame (df):

    print()
    print(df)

def print_metrics(model_accuracy, model_f1_score):

    print()
    print("Accuracy on Test Data:", model_accuracy)
    print("f1 score on Test Data:", model_f1_score)