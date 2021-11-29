import pickle, os

import pandas as pd
from skimage.transform import rescale
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt


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

    accuracy_validation, f1_validation, model_location, non_skipped_values, all_f1_score, all_roc_values = [], [], [], [], [], []

    for small_dataset in range(10, 110, 10):

        for hyperparameter_value in hyperparameter_list: 

            # Create a classifier: a support vector classifier
            if classifier == 'SVM':
                clf = svm.SVC(gamma = hyperparameter_value, probability = True)
            elif classifier == 'DecisionTree':
                clf = tree.DecisionTreeClassifier(max_depth = hyperparameter_value)

            # Learn the digits on the train subset
            X_train_smaller_set = X_train[0: int(len(X_train) * (small_dataset/100))]
            y_train_smaller_set = y_train[0: int(len(y_train) * (small_dataset/100))]
            clf.fit(X_train_smaller_set, y_train_smaller_set)

            # Predict the value of the digit on the test subset
            predicted_val = clf.predict(X_validation)

            val_accuracy_score = accuracy_score(y_validation, predicted_val)
            val_f1_score = f1_score(y_validation, predicted_val, average='weighted')

            #if val_accuracy_score < 0.11:
            #    print("Skipping for the hyperparameter value:", hyperparameter_value)
            #    continue

            #Save the model to disk
            saved_model = pickle.dumps(clf)
            model_path = '/home/niladri/mlops/mnist-example/models'
            path = model_path + '/' + classifier + '_' + str(hyperparameter_value) + '.pkl' 
            with open(path, 'wb') as f:
                pickle.dump(clf, f)

            
            accuracy_validation.append(val_accuracy_score)
            f1_validation.append(val_f1_score)
            model_location.append(path)
            non_skipped_values.append(hyperparameter_value)

        df = pd.DataFrame(data = {'Hyperparameter Value': non_skipped_values ,'Accuracy of Validation Data': accuracy_validation, 'f1 score of Validation Data': f1_validation, 'Model Location': model_location})
        best_model_path = get_best_model_path(df)
        best_model = load(best_model_path)
        
        predicted_test = best_model.predict(X_test)
        predicted_test_proba = best_model.predict_proba(X_test)
        test_f1_score = f1_score(y_test, predicted_test, average='macro')
        roc_auc_score_values = roc_auc_score(y_test, predicted_test_proba, multi_class="ovr")
        all_f1_score.append(test_f1_score)
        all_roc_values.append(roc_auc_score_values)


    make_plot_f1(all_f1_score)

    x_axis_value = [str(val - 10) + "-" + str(val)  for val in range(20, 110, 10)]
    y_axis_values = [all_roc_values[i+1] - all_roc_values[i] for i in range(9)]
    make_plot_roc(x_axis_value, y_axis_values)

    return df

def make_plot_roc(x_axis_value, y_axis_values):
    plt.figure()
    plt.plot(x_axis_value, y_axis_values)
    plt.xlabel("Smaller Training Set Percentage")
    plt.ylabel("ROC Score")
    plt.title('Smaller Training Set Percentage vs ROC Score')
    plt.savefig('/home/niladri/mlops/mnist-example/images/roc_plot.png', dpi=300, bbox_inches='tight')

def make_plot_f1(all_f1_score):
    plt.figure()
    plt.plot(range(10, 110, 10), all_f1_score)
    plt.xlabel("Smaller Training Set Percentage")
    plt.ylabel("Macro f1 score on Test set")
    plt.title('Smaller Percentage of Training set vs Macro f1 score on test set')
    plt.savefig('/home/niladri/mlops/mnist-example/images/line_plot.png', dpi=300, bbox_inches='tight')

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

def print_metrics(model_accuracy, model_f1_score):

    print()
    print("Accuracy on Test Data:", model_accuracy)
    print("f1 score on Test Data:", model_f1_score)

def load(model_path):
     return pickle.load(open(model_path, 'rb'))
     