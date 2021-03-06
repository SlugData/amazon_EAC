from sklearn import *
from sklearn.linear_model import *
import pandas as pd
import numpy as np

def main():
    # read in data, parse into training and target sets
    dataset = pd.read_csv('Data/train.csv')
    target = dataset.ACTION.values
    train = dataset.drop('ACTION', axis=1).drop('ROLE_ROLLUP_1',axis=1).drop('ROLE_FAMILY', axis=1).drop('ROLE_CODE', axis=1).values
    #train = dataset.drop('ACTION', axis=1).drop('ROLE_CODE',axis=1).values

    # Simple KFold cross validation. 10 folds.
    cv = cross_validation.KFold(train.shape[0], n_folds=10, indices=True)

    # iterate through the training and test cross validation segments and
    # run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:

        # Test/train data
        train_subset = train[traincv]
        test_subset = train[testcv]
        target_train_subset = target[traincv]
        target_test_subset = target[testcv]

        # One Hot Encode train and test data
        enc = preprocessing.OneHotEncoder()
        enc.fit(train)
        train_subset = enc.transform(train_subset)
        test_subset = enc.transform(test_subset)

        # Feature Selection
        # Logistic Regression for Feature Selection
        lrc = LogisticRegression(C=3.0)
        lrc.fit(train_subset, target_train_subset)
        train_subset = lrc.transform(train_subset)
        test_subset = lrc.transform(test_subset)

        # Feature Selection
        # Linear Support Vector Classification for Feature Selection
        #lsvc = svm.LinearSVC()
        #lsvc.fit(train_subset, target_train_subset)
        #train_subset = lsvc.transform(train_subset)
        #test_subset = lsvc.transform(test_subset)

        # Model Selection (will include Hyperparameter Optimization)
        cfr = LogisticRegression(C=3.0)

        # Predictions and Scoring
        predictions = cfr.fit(train_subset, target_train_subset).predict_proba(test_subset)[:,1]
        true_labels = target_test_subset
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label = 1)
        auc = metrics.auc(fpr, tpr)
        results.append(auc)

        print "Result: " + str(auc)

    print "Average Result: " + str( np.array(results).mean() )

if __name__=="__main__":
    main()
