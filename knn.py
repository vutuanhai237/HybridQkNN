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
        print(xs)
        print(ys)
        distances.append(fidelity(xs,ys))
    return distances




xs = [1,2,3,2]
yss = [
    [1,1,1,2],
    [1,0,0,2]
]


# indices_of_sorted_xs = sort_but_return_index(xs)
# labels = get_sublist_with_indices(ys, indices_of_sorted_xs, 4)
# print(major_vote(labels))
print(fidelity(xs, yss[0]))
print(distances(xs,yss))





