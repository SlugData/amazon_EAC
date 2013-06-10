from sklearn.ensemble import *
from sklearn import cross_validation
import pandas as pd
import numpy as np
from sklearn import metrics

def main():
    # read in data, parse into training and target sets
    dataset = pd.read_csv('Data/train.csv')
    target = dataset.ACTION.values
    train = dataset.drop('ACTION', axis=1).values

    # Use random forest classifier
    cfr = RandomForestClassifier(n_estimators=100)
    #cfr = GradientBoostingClassifier(n_estimators=1000, subsample = 0.1)

    # Simple KFold cross validation. 10 folds.
    cv = cross_validation.KFold(len(train), n_folds=10, indices=False)

    # iterate through the training and test cross validation segments and
    # run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        predictions = cfr.fit(train[traincv], target[traincv]).predict(train[testcv])
        true_labels = target[testcv]
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label = 1)
        auc = metrics.auc(fpr, tpr)
        results.append(auc)

    print "Results: " + str( np.array(results).mean() )

if __name__=="__main__":
    main()
