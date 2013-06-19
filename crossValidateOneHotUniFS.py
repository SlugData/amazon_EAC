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

    # Use Linear model classifier (Logistic Regression)
    cfr = LogisticRegression(C=3.0)

    # Simple KFold cross validation. 10 folds.
    cv = cross_validation.KFold(train.shape[0], n_folds=10, indices=True, shuffle=False)

    # iterate through the training and test cross validation segments and
    # run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:

        # Partion dataset into four parts
        train_subset = train[traincv]
        test_subset = train[testcv]
        target_train_subset = target[traincv]
        target_test_subset = target[testcv]

        # OneHotEncoding
        # Note for a transformed dataset, the ith column is the ith ACTIVE binary feature
        #enc = preprocessing.OneHotEncoder()
        #enc.fit(np.vstack((test_subset, train_subset)))
        train_subset = enc.transform(train_subset)
        test_subset = enc.transform(test_subset)

        # Feature selection via chi2
        low_pval_feature_indices = chi2_fs_feature_indices(train_subset, target_train_subset, 0.01)
        train_subset = train_subset[:, low_pval_feature_indices]
        test_subset = test_subset[:, low_pval_feature_indices]

        # Predictions and Scoring
        predictions = cfr.fit(train_subset, target_train_subset).predict_proba(test_subset)[:,1]
        true_labels = target_test_subset
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label = 1)
        auc = metrics.auc(fpr, tpr)
        results.append(auc)
        
        print "Result: " + str(auc)

    print "Average Result: " + str( np.array(results).mean() )


def chi2_fs_feature_indices(train, target, pval=0.05):
    # Feature selection via chi2
    # Takes training set train, target set target, and a pvalue threshold
    # Returns np.array of feature (i.e., column) indices whose pvalue < pval

    chi2_test_results = feature_selection.chi2(train, target)
    chi2_chi2_test = chi2_test_results[0]
    pval_chi2_test = chi2_test_results[1]

    low_pval_tf = pval_chi2_test < pval  # Set p-value threshold
    low_pval_feature_indices = []
    i = 0
    while i<len(pval_chi2_test):
        if low_pval_tf[i] == True:
            low_pval_feature_indices.append(i)
        i += 1
    low_pval_feature_indices = np.array(low_pval_feature_indices)

    return low_pval_feature_indices

    #print "chi2: " + str(chi2_chi2_test[low_pval_feature_indices])
    #print "p-values: " + str(pval_chi2_test[low_pval_feature_indices])
    #print "low_pval_features: " + str(low_pval_feature_indices)
    #print "Number of low_pval_features: " + str(len(low_pval_feature_indices))


if __name__=="__main__":
    main()
