import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# Get the pertinent data
# f=raw_input('Input the path to support vector metadata:     ')
f = 'Categories.csv'

data = np.genfromtxt(f, delimiter = ',')
datastring = np.genfromtxt(f, dtype = 'string', delimiter = ',')

print 'done with files'

'''
duration = data[:,5]
peak = data[:,9]
central = data[:,10]
bandwidth = data[:,11]
amplitude = data[:,13]
snr = data[:,14]
'''

# Support Vectors (X => vectors, y => classifiers)

X=data[:,[5,9]]
y=datastring[:,18]

h = 0.02   # step size in the mesh
C = 1.0    # SVM regularization parameter

# Multiple SVMs
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
# rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
# lin_svc = svm.LinearSVC(C=C).fit(X, y)

print 'Created classifiers...'
# Get mins/maxs

d_min, d_max = X[:, 0].min() - 1, X[:, 0].max() + 1
p_min, p_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# c_min, c_max = X[:, 2].min() - 2, X[:, 2].max() + 1
# b_min, b_max = X[:, 3].min() - 1, X[:, 3].max() + 1
# a_min, a_max = X[:, 4].min() - 1, X[:, 4].max() + 1
# s_min, s_max = X[:, 5].min() - 1, X[:, 5].max() + 1

# d=duration, p=peak, etc.



# dd = np.meshgrid(np.arange(d_min, d_max, h))
# pp = np.meshgrid(np.arange(p_min, p_max, h))
# cc = np.meshgrid(np.arange(c_min, c_max, h))
# bb = np.meshgrid(np.arange(b_min, b_max, h))
# aa = np.meshgrid(np.arange(a_min, a_max, h))
# ss = np.meshgrid(np.arange(s_min, s_max, h))
# dd=np.asarray(dd)
# pp=np.asarray(pp)
# cc=np.asarray(cc)
# bb=np.asarray(bb)
# aa=np.asarray(aa)
# ss=np.asarray(ss)



# Create mesh for things you are plotting
dd, pp = np.meshgrid(np.arange(d_min, d_max, h),
                     np.arange(p_min, p_max, h))


# Plotting duration vs amplitude for this instance
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']
          
print 'Entering for loop...'

i = 0
clf = svc
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
plt.subplot(2, 2, i + 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

Z = clf.predict(np.c_[dd.ravel(), pp.ravel()])

# Put the result into a color plot
Z = Z.reshape(dd.shape)
plt.contourf(dd, pp, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Duration')
plt.ylabel('Amplitude')
plt.xlim(dd.min(), dd.max())
plt.ylim(pp.min(), pp.max())
plt.xticks(())
plt.yticks(())
plt.title(titles[i])
print 'iterating...'
    
plt.show()
