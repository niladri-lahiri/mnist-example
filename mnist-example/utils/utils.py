import pickle, os

import pandas as pd
from skimage.transform import rescale
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree


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


def create_splits(data, target, test_size, validation_size_from_test_size):

    # Split data into 70% train and 15% validation and 15% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, shuffle=False)

    X_test, X_validation, y_test, y_validation = train_test_split(
        X_test, y_test, test_size=validation_size_from_test_size, shuffle=False)
    
    return X_train, X_test, X_validation, y_train, y_test, y_validation



def train_and_save_model(classifier, hyperparameter_list, X_train, y_train, X_validation, y_validation, X_test, y_test):

    accuracy_validation, f1_validation, model_location, non_skipped_values = [], [], [], []
    hyperparameter_val_1, hyperparameter_val_2 = [], []
    accuracy_train, f1_train = [], []
    accuracy_test, f1_test = [], []

    for hyperparameter_value in hyperparameter_list: 

        # Create a classifier: a support vector classifier
        if classifier == 'SVM':
            clf = svm.SVC(gamma = hyperparameter_value)
        elif classifier == 'DecisionTree':
            clf = tree.DecisionTreeClassifier(max_depth = hyperparameter_value[0], max_features = hyperparameter_value[1], criterion = hyperparameter_value[2])

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted_val = clf.predict(X_validation)

        val_accuracy_score = accuracy_score(y_validation, predicted_val)
        val_f1_score = f1_score(y_validation, predicted_val, average='weighted')

        # Predict the value of the digit on the test subset
        predicted_train = clf.predict(X_train)

        train_accuracy_score = accuracy_score(y_train, predicted_train)
        train_f1_score = f1_score(y_train, predicted_train, average='weighted')

        # Predict the value of the digit on the test subset
        predicted_test = clf.predict(X_test)

        test_accuracy_score = accuracy_score(y_test, predicted_test)
        test_f1_score = f1_score(y_test, predicted_test, average='weighted')

        #Save the model to disk
        saved_model = pickle.dumps(clf)
        model_path = '/home/niladri/mlops/mnist-example/models'
        path = model_path + '/' + classifier + '_' + str(hyperparameter_list[0]) + '_' + 'hyperparameter_list[2]' + '.pkl' 
        with open(path, 'wb') as f:
            pickle.dump(clf, f)

        
        accuracy_validation.append(val_accuracy_score)
        f1_validation.append(val_f1_score)

        accuracy_train.append(train_accuracy_score)
        f1_train.append(train_f1_score)

        accuracy_test.append(test_accuracy_score)
        f1_test.append(test_f1_score)

        print(f1_test)

        model_location.append(path)
        non_skipped_values.append(hyperparameter_value[0])
        hyperparameter_val_1.append(hyperparameter_value[1])
        hyperparameter_val_2.append(hyperparameter_value[2])

    #df = pd.DataFrame(data = {'Max Depth ': non_skipped_values , 'Max Features ': hyperparameter_val_1, 'Criterion ': hyperparameter_val_2, 'Accuracy of Train Data': accuracy_train, 'f1 score of Train Data': f1_train,  'Accuracy of Validation Data': accuracy_validation, 'f1 score of Validation Data': f1_validation,  'Accuracy of Test Data': accuracy_test, 'f1 score of Test Data': f1_test,'Model Location': model_location})

    #print(df)

    df = pd.DataFrame(data = {'Max Depth ': non_skipped_values , 'Max Features ': hyperparameter_val_1, 'Criterion ': hyperparameter_val_2, 'Accuracy of Train Data': accuracy_train, 'f1 score of Train Data': f1_train,  'Model Location': model_location})
    print(df)
    print()
    acc_df_mean = df['Accuracy of Train Data'].mean()
    f1_df_mean = df['f1 score of Train Data'].mean()
    print("Accuracy Mean of Train:", acc_df_mean)
    print("f1 score Mean of Train:", f1_df_mean)
    print()

    df = pd.DataFrame(data = {'Max Depth ': non_skipped_values , 'Max Features ': hyperparameter_val_1, 'Criterion ': hyperparameter_val_2, 'Accuracy of Validation Data': accuracy_validation, 'f1 score of Validation Data': f1_validation,  'Model Location': model_location})
    print(df)
    print()
    acc_df_mean = df['Accuracy of Validation Data'].mean()
    f1_df_mean = df['f1 score of Validation Data'].mean()
    print("Accuracy Mean of Validation:", acc_df_mean)
    print("f1 score Mean of Validation:", f1_df_mean)
    print()
    df = pd.DataFrame(data = {'Max Depth ': non_skipped_values , 'Max Features ': hyperparameter_val_1, 'Criterion ': hyperparameter_val_2, 'Accuracy of Test Data': accuracy_test, 'f1 score of Test Data': f1_test,  'Model Location': model_location})
    print(df)
    print()
    acc_df_mean = df['Accuracy of Test Data'].mean()
    f1_df_mean = df['f1 score of Test Data'].mean()
    print("Accuracy Mean of Test:", acc_df_mean)
    print("f1 score Mean of Test:", f1_df_mean)
    print()

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


def print_data_frame (clf_name, df):

    print()
    print("Metrics for:", clf_name)
    print(df)
    acc_df_mean = df['Accuracy of Validation Data'].mean()
    f1_df_mean = df['f1 score of Validation Data'].mean()
    print("Accuracy Mean of Validation:", acc_df_mean)
    print("f1 score Mean of Validation:", f1_df_mean)

def print_metrics(model_accuracy, model_f1_score):

    print()
    print("Accuracy on Test Data:", model_accuracy)
    print("f1 score on Test Data:", model_f1_score)

def load(model_path):
     return pickle.load(open(model_path, 'rb'))
     