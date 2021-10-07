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
rescale_factor = int(sys.argv[1])
gamma_values = map(float, sys.argv[2].strip('[]').split(','))
test_size = float(sys.argv[3])
validation_size_from_test_size = float(sys.argv[4])/float(sys.argv[3])

flattened_images, digits = flatten_images()
rescaled_images = preprocess(flattened_images, rescale_factor = rescale_factor)
X_train, X_test, X_validation, y_train, y_test, y_validation = create_splits(rescaled_images, digits, test_size, validation_size_from_test_size)
df = train_and_save_model(gamma_values, X_train, y_train, X_validation, y_validation)
model_path = get_best_model_path(df)
print_data_frame(df)
accuracy_test, f1_score_test = model_test(model_path, X_test, y_test)
print_metrics(accuracy_test, f1_score_test)
