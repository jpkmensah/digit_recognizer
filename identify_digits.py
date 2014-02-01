#-------------------------------------------------------------------------------
# Name:        identify_digits.py
# Purpose:     Identify a set of handwritten digits using an SVM
#
# Author:      Joshua Mensah
#
# Created:     30/01/2014
# Copyright:   (c) Joshua Mensah 2014
#-------------------------------------------------------------------------------
import numpy
import scipy
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import svm, metrics
import os

def main():
    predict_on_test('train.csv','test.csv','pred.csv')
    pass

def predict_on_test(training_filename, test_filename, pred_filename):
    train_set = numpy.loadtxt(training_filename, delimiter=',', skiprows=1)
    # first column represents the label for the values
    labels = train_set[:,0]
    n_examples = len(labels)
    values = scipy.delete(train_set,0,1)

    classifier = svm.SVC(C=1, gamma=0.001, degree=2, kernel='poly')
    classifier.fit(values, labels)

    test_values = numpy.loadtxt(test_filename, delimiter=',', skiprows=1)

    predicted = classifier.predict(test_values)

    predictions_file = open(pred_filename, 'w')
    img_id = 1
    predictions_file.write('ImageId,Label\n')
    for guess in predicted:
        output_line = "%d,%d\n" % (img_id, guess)
        predictions_file.write(output_line)
        img_id += 1
    predictions_file.close()


def split_train_cv(training_filename):
    # Split the training set into training and cross validation sets.
    # By default training set will be 75% and cross validation 25% of the
    # original training file by lines. Removes header file.
    training_file = open(training_filename, 'r')
    current_line = training_file.readline()
    line_count = 0
    while current_line != "":
        current_line = training_file.readline()
        line_count += 1
    training_file.seek(0)
    current_line = training_file.readline()
    split_line = line_count * .75
    (training_filename_short, training_filename_ext) = \
            os.path.splitext(training_filename)
    line_number = 0
    new_training_file = open(training_filename_short + \
            '_train' + training_filename_ext, 'w')
    new_cv_file = open(training_filename_short + \
            '_cv' + training_filename_ext, 'w')
    while current_line != "":
        current_line = training_file.readline()
        if line_number < split_line:
            new_training_file.write(current_line)
        else:
            new_cv_file.write(current_line)
        line_number += 1
    training_file.close()
    new_training_file.close()
    new_cv_file.close()

def test_on_cv(training_filename, cv_filename):
    training_set = numpy.loadtxt(training_filename, delimiter=',')
    training_labels = training_set[:,0]
    n_examples = len(labels)
    training_values = scipy.delete(training_set,0,1)

    classifier = svm.SVC(C=1, gamma=0.001, degree=2, kernel='poly')
    classifier.fit(values, labels)

    cv_set = numpy.loadtxt(cv_filename, delimiter=',')
    cv_labels = cv_set[:,0]
    cv_values = scipy.delete(cv_set,0,1)

    cv_predicted = classifier.predict(cv_values)

    print("Classification report for classifier %s:\n%s\n" % \
            (classifier, metrics.classification_report(cv_labels,predicted)))
    print("Confusion matrix:\n%s" % \
            metrics.confusion_matrix(cv_labels, predicted))

if __name__ == '__main__':
    main()
