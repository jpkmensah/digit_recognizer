#-------------------------------------------------------------------------------
# Name:        identify_digits.py
# Purpose:
#
# Author:      Joshua Mensah
#
# Created:     30/01/2014
# Copyright:   (c) Joshua Mensah 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy
import scipy
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import svm, metrics
import cPickle as pickle

def main():
    a = numpy.loadtxt('train_m.csv', delimiter=',', skiprows=1)
    # first column represents the label for the values
    labels = a[:,0]
    n_examples = len(labels)
    values = scipy.delete(a,0,1)

    classifier = svm.SVC(C=1, gamma=0.001, degree=2, kernel='poly')
    classifier.fit(values, labels)

    b = numpy.loadtxt('cv_m.csv', delimiter=',', skiprows=1)
    cv_labels = b[:,0]
    cv_values = scipy.delete(b,0,1)

    expected = cv_labels
    predicted = classifier.predict(cv_values)

    # pickle.dump(classifier, open("classifier.p", "wb"))

    '''
    for index, (image, expected, prediction) in enumerate(zip(cv_values, expected, predicted)[:4]):
        # cv_square = cv_values[1].reshape((28,28))
        pl.subplot(1,3, index + 1)
        cv_square = image.reshape((28,28))
        pl.imshow(cv_square, cmap=pl.cm.gray_r, interpolation='nearest')
        # pl.title('Training: %i Prediction: %i' % (cv_labels[1], predicted[1]))
        pl.title('T: %i P: %i' % (expected, prediction))

    pl.show()
    '''

    print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected,predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    pass

def split_train_cv():
    print "Splitting original test set into test and cross validation"
    f = open('train.csv')
    i = 0
    r = f.readline()
    testfile = open('train_m.csv', 'w+')
    cvfile = open('cv_m.csv', 'w+')
    while r != "":
        r = f.readline()
        if i < 30000:
            testfile.write(r)
        else:
            cvfile.write(r)
        i += 1
    # a = numpy.loadtxt('test.csv')
    f.close()
    testfile.close()
    cvfile.close()

if __name__ == '__main__':
    main()
