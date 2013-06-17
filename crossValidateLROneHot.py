from sklearn import *
from sklearn.linear_model import *
import pandas as pd
import numpy as np

def main():
    # read in data, parse into training and target sets
    dataset = pd.read_csv('Data/train.csv')
    target = dataset.ACTION.values
    train = dataset.drop('ACTION', axis=1).drop('ROLE_ROLLUP_1',axis=1).drop('ROLE_FAMILY', axis=1).drop('ROLE_CODE', axis=1).values

    # OneHotEncoding
    enc = preprocessing.OneHotEncoder()
    enc.fit(train)
    train = enc.transform(train)

    # Use Linear model classifier (Logistic Regression)
    cfr = LogisticRegression(C=3.0)

    # Simple KFold cross validation. 10 folds.
    #cv = cross_validation.KFold(len(train), n_folds=10, indices=False)
    cv = cross_validation.KFold(train.shape[0], n_folds=10, indices=True)

    # iterate through the training and test cross validation segments and
    # run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        predictions = cfr.fit(train[traincv], target[traincv]).predict_proba(train[testcv])[:,1]
        #predictions = cfr.fit(train[traincv], target[traincv]).predict(train[testcv])
        true_labels = target[testcv]
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label = 1)
        auc = metrics.auc(fpr, tpr)
        results.append(auc)

    for r in results:
        print "Result: " + str(r)
    print "Average Result: " + str( np.array(results).mean() )

if __name__=="__main__":
    main()
