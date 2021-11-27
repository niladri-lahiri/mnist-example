"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

import sys
from utils.utils import flatten_images, preprocess, create_splits, train_and_save_model, get_best_model_path, model_test, print_data_frame, print_metrics

#Get values from command line
rescale_factor = 1
#gamma_values = map(float, sys.argv[2].strip('[]').split(','))
hyperparameters = [[ 16, 'log2', 'gini'], [16, 'log2', 'gini'], [16, 'log2', 'gini']]
#hyperparameters = [[ 4, 'sqrt', 'gini'], [8, 'auto', 'entropy'], [16, 'log2', 'gini']]
test_size = 0.3
validation_size_from_test_size = 0.15

flattened_images, digits = flatten_images()
rescaled_images = preprocess(flattened_images, rescale_factor = rescale_factor)
X_train, X_test, X_validation, y_train, y_test, y_validation = create_splits(rescaled_images, digits.target, test_size, validation_size_from_test_size)
#classifiers = {'SVM' : gamma_values, 'DecisionTree' : depth_value}
classifiers = {'DecisionTree' : hyperparameters}
for clf_name in classifiers:
    df = train_and_save_model(clf_name, classifiers[clf_name], X_train, y_train, X_validation, y_validation, X_test, y_test)

    #model_path = get_best_model_path(df)
    #print_data_frame(clf_name, df)
    #accuracy_test, f1_score_test = model_test(model_path, X_test, y_test)
    #print_metrics(accuracy_test, f1_score_test)
