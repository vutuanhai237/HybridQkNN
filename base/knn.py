import numpy as np
import collections, base.swaptest, sklearn

def encode(xss):
    """Convert normal vector to normalized state

    Args:
        xss (list of list): dataset

    Returns:
        list of list: normalized dataset
    """
    amplitudes = np.sqrt(np.einsum('ij,ij->i', xss, xss))
    amplitudes[amplitudes == 0] = 1
    normalised_data = xss / amplitudes[:, np.newaxis]
    return normalised_data
def get_major_vote(labels):
    """Get major label value

    Args:
        labels (list): list of value

    Returns:
        int: major vote
    """
    x = collections.Counter(labels)
    return x.most_common(1)[0][0]

def sort_return_index(xs):
    """Note that we must sort for large to small, because scalar product between 
    two similar vectors is almost 1 (other wise with Eucliean distance), I wrong this
    point and take few hours to catch this error :))

    Args:
        xs (list): list of scalar products between train vectors and one test vector

    Returns:
        list: sorted list but return indices 
    """
    new_xs = sorted(range(len(xs)), key=lambda k: xs[k])
    new_xs.reverse()
    return new_xs

def get_sublist_with_indices(xs, indices, k):
    return [xs[index] for index in indices][:k]

def distances(xs, yss, iteration: int = 1):
    """Return a lots of distance

    Args:
        - xs (list of float): vector
        - yss (list of list): dataset
        - iteration (int): number of iteration

    Returns:
        - list of values: all distances from vector to others vector in dataset
    """
    distances = []
    for ys in yss:
        distances.append(base.swaptest.get_fidelity(vector1 = xs, vector2 = ys, iteration = iteration))
    return distances

def predict(train_datas, train_labels, test_datas, k: int = 1, iteration: int = 1):
    """Return predicted labels QKNN algorithm

    Args:
        - train_datas (numpy array 2D): Vectors in train data
        - train_labels (numpy array 1D): Labels in train data
        - test_datas (numpy array 2D): Vectors in test data
        - k (int): Number of neighboors
        - iteration (int): number of iteration
    Returns:
        - list of int: predicted labels
    """
    predict_labels = []
    i = 0
    for i, test_data in enumerate(test_datas):
        xs = distances(test_data, train_datas, iteration)
        indices_of_sorted_xs = sort_return_index(xs)
        labels = get_sublist_with_indices(train_labels, indices_of_sorted_xs, k)
        predict_labels.append(get_major_vote(labels))
        if i % 10 == 0:
            print("Progress " + (str(int(i/len(test_datas)*100)) + '%'))
    return predict_labels

def bench_mark(ground_truth, predict):
    """Return predict labels QKNN algorithm

    Args:
        - ground_truth (numpy array 1D): truth labels
        - predict (numpy array 1D): predict labels

    Returns:
        - Tuple: benchmark on classifer problem
    """
    accuracy = sklearn.metrics.accuracy_score(ground_truth, predict)
    precision = sklearn.metrics.precision_score(ground_truth, predict, average="weighted")
    recall = sklearn.metrics.recall_score(ground_truth, predict, average="weighted")
    f1 = sklearn.metrics.f1_score(ground_truth, predict, average="micro")
    matrix = sklearn.metrics.confusion_matrix(ground_truth, predict)
    return accuracy, precision, recall, matrix