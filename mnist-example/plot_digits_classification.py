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
rescale_factor = 1 #int(sys.argv[1])
gamma_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] #map(float, sys.argv[2].strip('[]').split(','))
#depth_value = map(float, sys.argv[3].strip('[]').split(','))
test_size = 0.2 #float(sys.argv[4])
validation_size_from_test_size = 0.5 #float(sys.argv[5])/float(sys.argv[4])

flattened_images, digits = flatten_images()
rescaled_images = preprocess(flattened_images, rescale_factor = rescale_factor)
X_train, X_test, X_validation, y_train, y_test, y_validation = create_splits(rescaled_images, digits.target, test_size, validation_size_from_test_size)
#classifiers = {'SVM' : gamma_values, 'DecisionTree' : depth_value}
classifiers = {'SVM' : gamma_values}
for clf_name in classifiers:
    df = train_and_save_model(clf_name, classifiers[clf_name], X_train, y_train, X_validation, y_validation, X_test, y_test)
    model_path = get_best_model_path(df)
    print_data_frame(clf_name, df)
    accuracy_test, f1_score_test = model_test(model_path, X_test, y_test)
    print_metrics(accuracy_test, f1_score_test)
