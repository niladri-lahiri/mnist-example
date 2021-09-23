from skimage.transform import rescale
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

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


def test(clf, X, y):
    predicted_test = clf.predict(X)

    accuracy_test = accuracy_score(y, predicted_test)
    f1_score_test = f1_score(y, predicted_test, average='weighted')

    return accuracy_test, f1_score_test