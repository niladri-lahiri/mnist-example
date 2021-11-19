import sys, os, math, random
from sklearn import datasets

#To find the utils.utils package
testdir = os.path.dirname(__file__)
print(testdir)
srcdir = '../mnist-example'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

from utils.utils import flatten_images, preprocess, create_splits, train_and_save_model, get_best_model_path, get_best_model_metrics, model_test

sys.path.insert(1, '/home/niladri/mlops/mnist-example/mnist-example/utils')
import utils

clf_best_svm = utils.utils.load('/home/niladri/mlops/mnist-example/models/SVM_0.001.pkl')
clf_best_decision_tree = utils.utils.load('/home/niladri/mlops/mnist-example/models/DecisionTree_18.0.pkl')


def test_equality():

    assert 1==1


def test_model_writing():

    gamma, rescale_factor, test_size, validation_size = [0.001], 1, 0.3, 0.15
    validation_size_from_test_size = float(validation_size)/float(test_size)

    flattened_images, digits = flatten_images()
    rescaled_images = preprocess(flattened_images, rescale_factor = rescale_factor)
    X_train, X_test, X_validation, y_train, y_test, y_validation = create_splits(rescaled_images, digits.target, test_size, validation_size_from_test_size)
    df = train_and_save_model('SVM', gamma, X_train, y_train, X_validation, y_validation)
    model_path = get_best_model_path(df)

    assert os.path.isfile(model_path)


def test_small_data_overfit_checking():

    gamma, rescale_factor, test_size, validation_size, subsampling, threshold = [0.001], 1, 0.3, 0.15, 125, 0.05
    validation_size_from_test_size = float(validation_size)/float(test_size)

    flattened_images, digits = flatten_images()
    rescaled_images = preprocess(flattened_images, rescale_factor = rescale_factor)
    X_train, X_test, X_validation, y_train, y_test, y_validation = create_splits(rescaled_images, digits.target, test_size, validation_size_from_test_size)
    df = train_and_save_model('SVM', gamma, X_train[:subsampling], y_train[:subsampling], X_train[:subsampling], y_train[:subsampling])
    model_path = get_best_model_path(df)
    accuracy_train, f1_score_train = get_best_model_metrics(df)
    accuracy_test, f1_score_test = model_test(model_path, X_test, y_test)

    assert accuracy_train > accuracy_test + threshold and f1_score_train > f1_score_test + threshold


def test_create_split_100_samples():

    rescale_factor, test_size, validation_size, subsampling = 1, 0.3, 0.2, 100
    validation_size_from_test_size = validation_size/test_size

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target 

    picking_100_indices = random.choices(range(len(X)), k = subsampling)
    X_sampled, y_sampled =  X[picking_100_indices], y[picking_100_indices]
    X_train, X_test, X_validation, y_train, y_test, y_validation = create_splits(X_sampled, y_sampled, test_size, validation_size_from_test_size)

    assert len(X_train) == 70
    assert len(X_validation) == 21 #rounding off issue
    assert len(X_test) == 9 #rounding off issue
    assert len(X_train) + len(X_validation) + len(X_test) == 100



def test_create_split_9_samples():

    rescale_factor, test_size, validation_size, subsampling = 1, 0.3, 0.2, 9
    validation_size_from_test_size = float(validation_size)/float(test_size)

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target 

    picking_10_indices = random.choices(range(len(X)), k = subsampling)
    X_sampled, y_sampled =  X[picking_10_indices], y[picking_10_indices]
    X_train, X_test, X_validation, y_train, y_test, y_validation = create_splits(X_sampled, y_sampled, test_size, validation_size_from_test_size)

    assert len(X_train) == 6
    assert len(X_validation) == 2 
    assert len(X_test) == 1 
    assert len(X_train) + len(X_validation) + len(X_test) == 9

def test_digit_correct_0():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))

    assert clf_best_svm.predict(X)[0] == 0
    assert clf_best_decision_tree.predict(X)[0] == 0

def test_digit_correct_1():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))

    assert clf_best_svm.predict(X)[1] == 1
    assert clf_best_decision_tree.predict(X)[1] == 1

def test_digit_correct_2():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))

    assert clf_best_svm.predict(X)[2] == 2
    assert clf_best_decision_tree.predict(X)[2] == 2

def test_digit_correct_3():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    assert clf_best_svm.predict(X)[3] == 3
    assert clf_best_decision_tree.predict(X)[3] == 3

def test_digit_correct_4():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    assert clf_best_svm.predict(X)[4] == 4
    assert clf_best_decision_tree.predict(X)[4] == 4

def test_digit_correct_5():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    assert clf_best_svm.predict(X)[15] == 5
    assert clf_best_decision_tree.predict(X)[15] == 5

def test_digit_correct_6():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    assert clf_best_svm.predict(X)[6] == 6
    assert clf_best_decision_tree.predict(X)[6] == 6

def test_digit_correct_7():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    assert clf_best_svm.predict(X)[7] == 7
    assert clf_best_decision_tree.predict(X)[7] == 7

def test_digit_correct_8():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    assert clf_best_svm.predict(X)[8] == 8
    assert clf_best_decision_tree.predict(X)[8] == 8

def test_digit_correct_9():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))

    assert clf_best_svm.predict(X)[9] == 9
    assert clf_best_decision_tree.predict(X)[9] == 9

