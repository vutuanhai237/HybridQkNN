import numpy as np
from collections import Counter
from swaptest import fidelity
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
def major_vote(labels):
    x = Counter(labels)
    return x.most_common(1)[0][0]
def sort_but_return_index(xs):
    return sorted(range(len(xs)), key=lambda k: xs[k])
def get_sublist_with_indices(xs, indices, k):
    return [xs[index] for index in indices][:k]
def encode(xss):
    amplitudes = np.sqrt(np.einsum('ij,ij->i', xss, xss))
    amplitudes[amplitudes == 0] = 1
    normalised_data = xss / amplitudes[:, np.newaxis]
    return normalised_data
def distances(xs, yss):
    distances = []
    for ys in yss:
        distances.append(fidelity(xs,ys))
    return distances

def predict(train_datas, train_labels, test_datas, k):
    """Return predict labels QKNN algorithm

    Args:
        train_datas (numpy array 2D): Vectors in train data
        train_labels (numpy array 1D): Labels in train data
        test_datas (numpy array 2D): Vectors in test data
        k (interger): Number of neighboors

    Returns:
        [type]: [description]
    """
    predict_labels = []
    for test_data in test_datas:
        xs = distances(test_data, train_datas)
        indices_of_sorted_xs = sort_but_return_index(xs)
        labels = get_sublist_with_indices(train_labels, indices_of_sorted_xs, k)
        predict_labels.append(major_vote(labels))
    return predict_labels
    

# # xs = np.asarray([1,2,3,2])
# train_datas = np.asarray([
#     [1,1,1,2],
#     [1,0,0,2],
#     [1,2,0.5,2],
#     [1,0,0,0.6],
#     [2,3,0.5,2],
#     [2,0,0,0.6]
# ])

# train_labels = np.asarray([
#     1,2,1,1,0,2
# ])

# test_datas = np.asarray([
#     [1,1,1,2.1],
#     [1.3,0,0,2],
# ])

# print(predict(train_datas, train_labels, test_datas))





