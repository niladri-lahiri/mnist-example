
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

# Standard scientific Python imports
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from skimage.transform import rescale
from utils.utils import preprocess, create_splits, test


###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

gamma_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
#rescale_factor = [0.25, 0.5, 0.75, 1]
rescale_factor = [1]

accuracy_validation, accuracy_test = [], []
f1_validation, f1_test = [], []
model_location, non_skipped_gamma_values = [], []

for gamma in gamma_values: 
    

    images = preprocess(digits.images, rescale_factor = rescale_factor)

    test_size = 0.3
    validation_size_from_test_size = 0.5

    X_train, X_test, X_validation, y_train, y_test, y_validation = create_splits(data, digits, test_size, validation_size_from_test_size)

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=gamma)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted_val = clf.predict(X_validation)
    #predicted_test = clf.predict(X_test)

    ###############################################################################
    # Below we visualize the first 4 test samples and show their predicted
    # digit value in the title.

    '''
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted_test):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f'Prediction: {prediction}')
        '''

    ###############################################################################
    # :func:`~sklearn.metrics.classification_report` builds a text report showing
    # the main classification metrics.

    #Commenting
    #print(f"Classification report for classifier {clf}:\n"
    #    f"{metrics.classification_report(y_test, predicted_test)}\n")

    ###############################################################################
    # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
    # true digit values and the predicted digit values.

    #Commenting
    disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    #print(f"Confusion matrix:\n{disp.confusion_matrix}")

    val_accuracy_score = accuracy_score(y_validation, predicted_val)
    val_f1_score = f1_score(y_validation, predicted_val, average='weighted')

    if val_accuracy_score < 0.11:
        print("Skipping for gamma:", gamma)
        continue

    #Save the model to disk
    saved_model = pickle.dumps(clf)
    path = '/home/niladri/mlops/mnist-example/models/' + str(gamma) + '.pkl' 
    with open(path, 'wb') as f:
        pickle.dump(clf, f)
    
    accuracy_validation.append(val_accuracy_score)
    f1_validation.append(val_f1_score)
    model_location.append(path)
    non_skipped_gamma_values.append(gamma)


    #plt.show()

#print(pd.DataFrame(data = {'Gamma Value': gamma_values ,'Accuracy of Validation Data': accuracy_validation, 'Accuracy of Test Data': accuracy_test}))
#print()
#print("The best value of gamma is:", gamma_values[accuracy_test.index(max(accuracy_test))])

print()
print(pd.DataFrame(data = {'Gamma Value': non_skipped_gamma_values ,'Accuracy of Validation Data': accuracy_validation, 'f1 score of Validation Data': f1_validation, 'Model Location': model_location}))
best_model_path = model_location[accuracy_validation.index(max(accuracy_validation))]
best_gamma_value = non_skipped_gamma_values[accuracy_validation.index(max(accuracy_validation))]

print()
print("The best value of gamma is:", best_gamma_value)

loaded_model = pickle.load(open(best_model_path, 'rb'))

accuracy_test, f1_score_test = test(loaded_model, X_test, y_test)

print()
print("Best Model's accuracy on Test Data: ", accuracy_test)
print("Best Model's f1-score on Test Data: ", f1_score_test)
print()


