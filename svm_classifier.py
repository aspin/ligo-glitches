from sklearn import svm, manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

DEFAULT_PATH = 'Categories.csv'
RELEVANT_ATTRS = [5,9,10,11,13,14]

res_map = {
    'Null': 0,
    'RF Oscillator': 1,
    'IPC Error': 2
}

def create_vectors(path):
    X = np.genfromtxt(path, delimiter=',')[:, RELEVANT_ATTRS]
    Y = np.genfromtxt(path, dtype='string', delimiter=',')[:, 18]
    y = np.arange(Y.size)
    for i in range(len(Y)):
        y[i] = res_map[Y[i]]

    return X, y

def euclidean_transform(vector):
    differences = euclidean_distances(vector)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9)
    pos = mds.fit(differences).embedding_

    clf = PCA(n_components=2)
    return clf.fit_transform(pos)

def scale(vector):
    return vector / vector.max()

def svc_fit(X, Y):
    clf = svm.SVC()
    return clf.fit(X, Y)

def plot(X, y, clf):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.subplot(2, 2, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    plt.xlabel('Transformed X')
    plt.ylabel('Transformed Y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('Euclidean Transformation')
    return

X, Y = create_vectors(DEFAULT_PATH)
X = euclidean_transform(X)
X = scale(X)
clf = svc_fit(X, Y)
plot(X, Y, clf)

# fig = plt.figure(1)
# ax = plt.axes([0., 0., 1., 1.])

# plt.scatter(pos[:, 0], pos[:, 1], s=20, c='g')

# differences = differences.max() / differences * 100
# differences[np.isinf(differences)] = 0

# plt.show()

