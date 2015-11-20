from sklearn import svm, datasets, cross_validation, preprocessing
from random import shuffle, sample

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

DATA_FILE = 'Categories.csv'

Y_MAP = {
    'RF Oscillator': 0,
    'IPC Error': 1,
    'Null': 2,
    'Bird': 3,
    'Box': 4,
    'Low-Freq Tower': 5,
    'Mid-Freq Tower': 6,
    'Caterpillar': 7,
    'Comet': 8,
    'Forest': 9,
    'Hill': 10,
    'Mountains': 11,
    'Repeated Spikes': 12,
    'Scatter': 13,
    'Skinny Spike': 14,
    'Spike': 15,
    'Steps': 16, 
    'Stones': 17
}

Y_VALS = [
    'RF Oscillator',
    'IPC Error',
    'Null',
    'Bird',
    'Box',
    'Low-Freq Tower',
    'Mid-Freq Tower',
    'Caterpillar',
    'Comet',
    'Forest',
    'Hill',
    'Mountains',
    'Repeated Spikes',
    'Scatter',
    'Skinny Spike',
    'Spike',
    'Steps' ,
    'Stones'
]

# Attribute Reference
# [5,9,10,11,13,14] 
# [duration, peak, central, bandwidth, amplitude, SNR] 

def main():

    for i, name in zip([5,9,10,11,13], ['duration', 'peak', 'central', 'bandwidth', 'amplitude']):
        X, y = split_attributes(load_subset(DATA_FILE, i, 14))

        svc = svm.SVC(kernel='linear', C=1.0).fit(X, y)
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1.0).fit(X, y)
        poly_svc = svm.SVC(kernel='poly', degree=3, C=1.0).fit(X, y)
        lin_svc = svm.LinearSVC(C=1.0).fit(X, y)

        title = 'SNR vs. {0}'.format(name)
        plot(X, y, svc, name, 'SNR', 'SVC ' + title)
        plot(X, y, rbf_svc, name, 'SNR', 'RBF ' + title)
        plot(X, y, poly_svc, name, 'SNR', 'Poly ' + title)
        plot(X, y, lin_svc, name, 'SNR', 'LinearSVC ' + title)


    # X, y = split_attributes(load(DATA_FILE))

    # scipy cross validation
    # clf = svm.LinearSVC()
    # scores = cross_validation.cross_val_score(clf, X, y)
    # print scores
    
    # custom_cross_validate(DATA_FILE, 40, 5)

    # plot(X, y, clf)

def load(f):
    data = np.genfromtxt(f, delimiter = ',')
    datastring = np.genfromtxt(f, dtype = 'string', delimiter = ',')

    X = preprocessing.scale(data[:, [5, 9, 10, 11, 13, 14]])
    # X = preprocessing.scale(data[:, [14, 9]])
    # X = data[:, [5, 9]]

    # X = data[:, [5, 9, 10, 11, 13, 14]]
    y = datastring[:, 18]
    for i in xrange(len(y)):
        # print y[i]
        y[i] = Y_MAP[y[i]]

    output = []
    for i in xrange(len(X)):
        line = []
        for j in X[i]:
            line.append(j)
        line.append(int(y[i]))
        # line.append(y[i])
        output.append(line)

    return output

def load_subset(f, x, y):
    data = np.genfromtxt(f, delimiter = ',')
    datastring = np.genfromtxt(f, dtype = 'string', delimiter = ',')

    X = preprocessing.scale(data[:, [x, y]])
    y = datastring[:, 18]
    for i in xrange(len(y)):
        # print y[i]
        y[i] = Y_MAP[y[i]]

    output = []
    for i in xrange(len(X)):
        line = []
        for j in X[i]:
            line.append(j)
        line.append(int(y[i]))
        # line.append(y[i])
        output.append(line)

    return output

# custom cross validation -- I wasn't totally sure what the scipy one as doing
# so I wrote another one -- they perform similarly though
def custom_cross_validate(f, size, attempts):
    masterX, masterY = split_attributes(load(f))

    count = 0
    while count < attempts:
        currentX, currentY = split_attributes(random_balanced_load(f, size))
        # clf = svm.SVC(kernel='linear').fit(currentX, currentY)
        clf = svm.LinearSVC().fit(currentX, currentY)

        correct = 0
        incorrect = 1
        for i in xrange(len(masterX)):
            if clf.predict(masterX[i])[0] == masterY[i]:
                correct += 1
            else:
                incorrect += 1

        total = correct + incorrect
        print 'Run {0}: {1} correct, {2} incorrect, {3} percent accurate.'.format(count + 1, correct, incorrect, correct * 1.0 / total)
        count += 1

    return

# sample a random subset from the whole dataset in the same
# ratio as the examples are currently distributed, making
# sure to take into account at least one of each type
def random_balanced_load(f, size):
    data = load(f)
    balance, total = enumerate_counts(f)

    ratios = {}
    for key, val in balance.iteritems():
        ratios[key] = max(val * size / total, 1)

    index = 0
    output = []
    while index < len(data):
        current = data[index][6]
        total_current = balance[current]
        count = ratios[current]

        output += sample(data[index:index + total_current], count)
        index += total_current

    return output

def enumerate_counts(f):
    datastring = np.genfromtxt(f, dtype = 'string', delimiter = ',')
    y = datastring[:, 18]
    possible = list(set(y))

    counts = [[attr, 0] for i, attr in enumerate(possible)]
    for i in datastring:
        index = possible.index(i[18])
        counts[index][1] += 1

    return dict(counts), sum(map(lambda x: x[1], counts))

def split_attributes(data):
    return map(lambda x: x[:len(x) - 1], data), map(lambda y: y[len(y)-1], data)

def plot(X, y, clf, xlabel, ylabel, title):
    X = np.array(X)
    y = np.array(y)

    h = 0.2 # mesh stepsize
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    

    # Plot also the training points
    colors = plt.cm.rainbow(np.linspace(0, 1, 18))
    prev_range = 0
    curr_range = 0
    for i, c in zip(range(18), colors):
        prev_range = curr_range
        while (curr_range < len(y) and y[curr_range] == i):
            curr_range += 1

        if (i >= 0):
            plt.scatter(
                X[prev_range:curr_range, 0],
                X[prev_range:curr_range, 1], color=c, alpha=0.9, label=Y_VALS[i])

    plt.contourf(xx, yy, Z, colors=colors, alpha=0.8)


    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(())
    plt.yticks(())

    plt.title(title)
    # plt.savefig('svms/' + title + '.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    # plt.show()

main()

X, y = split_attributes(load(DATA_FILE))
