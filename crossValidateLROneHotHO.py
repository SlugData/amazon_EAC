from sklearn import *
from sklearn.linear_model import *
import pandas as pd
import numpy as np

def main():
    # read in data, parse into training and target sets
    dataset = pd.read_csv('Data/train.csv')
    target = dataset.ACTION.values
    train = dataset.drop('ACTION', axis=1).drop('ROLE_CODE', axis=1).values

    # OneHotEncoding
    enc = preprocessing.OneHotEncoder()
    enc.fit(train)
    train = enc.transform(train)

    # Hyperparameter Optimization via GridSearch
    lrc = LogisticRegression()
    parameters = { 'penalty':['l2'],'dual':[True,False], 'C':[1.0, 3.0, 5.0], 'fit_intercept':[True,False]}
    cfr = grid_search.GridSearchCV(lrc, parameters, score_func = metrics.auc_score, cv=None)

    # Use Linear model classifier (Logistic Regression)
    #cfr = LogisticRegression()

    # Simple KFold cross validation. 10 folds.
    cv = cross_validation.KFold(train.shape[0], n_folds=10, indices=True, shuffle=True)
    #cfr = grid_search.GridSearchCV(lrc, parameters, cv)

    # iterate through the training and test cross validation segments and
    # run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        predictions = cfr.fit(train[traincv], target[traincv]).predict_proba(train[testcv])[:,1]
        #predictions = cfr.fit(train[traincv], target[traincv]).predict(train[testcv])
        true_labels = target[testcv]
        #fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label = 1)
        #auc = metrics.auc(fpr, tpr)

        auc = metrics.auc_score(true_labels, predictions)
        results.append(auc)
       
        print cfr.best_estimator_
        print auc

    for r in results:
        print "Result: " + str(r)
    print "Average Result: " + str( np.array(results).mean() )

if __name__=="__main__":
    main()
