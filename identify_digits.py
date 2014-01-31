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

def main():
    a = numpy.loadtxt('train_m.csv', delimiter=',')
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
