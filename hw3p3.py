import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib as plt
import itertools
from sklearn.cross_validation import StratifiedKFold
#from __future__ import division
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from collections import defaultdict
from glob import glob
import sys
import math
import os

def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))


#Read Datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y_train=train["ACTION"]
target='ACTION'
ID=test['id']
test=test.drop(['id'],axis=1)
train['combination']=train['MGR_ID']*10000 + train['RESOURCE']
test['combination']=test['MGR_ID']*10000 + test['RESOURCE']
predictors = [x for x in train.columns if x not in target]
#Value_Counts 2-way
num=len(train.index)
way2iter=set(train.columns).difference(set(['ACTION', 'combination']))
way1iter=set(train.columns).difference(set(['ACTION']))
for i,j in itertools.combinations(way2iter,2):
    new=dict(train.groupby([i,j])[j].count())
    train[i+j]=pd.Series(zip(train[i],train[j])).map(new)
    test[i+j]=pd.Series(zip(test[i],test[j])).map(new)
#train.head()
way3lis=[("RESOURCE", "MGR_ID", "ROLE_ROLLUP_2"), 
("RESOURCE", "MGR_ID", "ROLE_DEPTNAME"),
("RESOURCE", "MGR_ID", "ROLE_CODE"),
("RESOURCE", "MGR_ID", "ROLE_FAMILY_DESC"),
("RESOURCE", "MGR_ID", "ROLE_FAMILY"),
("ROLE_FAMILY", "MGR_ID", "ROLE_DEPTNAME"),
("ROLE_FAMILY", "MGR_ID", "ROLE_ROLLUP_2"),
("RESOURCE", "ROLE_DEPTNAME", "ROLE_FAMILY"),
("RESOURCE", "ROLE_DEPTNAME", "ROLE_ROLLUP_2")]
#3way
for (i,j,k) in way3lis:
    new=dict(train.groupby([i,j,k])[j].count())
    train[i+j+k]=pd.Series(zip(train[i],train[j],train[k])).map(new)
    test[i+j+k]=pd.Series(zip(test[i],test[j],test[k])).map(new)
#Value_Counts 1-way
for i in way1iter:
    counts_tot=dict(train[i].value_counts())
    train[i+'count']=train[i].map(counts_tot)
    test[i+'count']=test[i].map(counts_tot)
    test=test.fillna(value=0)

rows=train.shape[0]
train=train.drop('ACTION',axis=1)
result = pd.concat([train,test])
for dfgs in predictors:
    affiliate_channel_maxs = []
    affiliate_channel_maxs_dict = {}
    affiliate_channel_maxs = list(enumerate(np.unique(result[dfgs])))
    affiliate_channel_maxs_dict = {name : i for i, name in affiliate_channel_maxs}
    result[dfgs+'_enum'] = result[dfgs].map(lambda x: affiliate_channel_maxs_dict[x]).astype(int)
new_train=result[:rows]
new_train[target]=y_train
new_test=result[rows:]

predictors = [x for x in new_train.columns if x not in target]

#new_train.to_csv('fff.csv')
#


X = new_train.as_matrix(predictors)
y = new_train.as_matrix(['ACTION']).ravel()

X_submission = new_test.as_matrix(predictors)
y_id = ID.as_matrix(['id']).ravel()

print X.shape

#Blending Starts

np.random.seed(0) # seed to shuffle the train set

n_folds = 15
verbose = True
shuffle = True

skf = StratifiedKFold(y, n_folds)
    
xgb_model = xgb.XGBClassifier(objective="binary:logistic",
                      learning_rate =0.1,
                      scale_pos_weight=1,
                      min_child_weight=6,
                      max_depth=20,
                      n_estimators=110,
                      colsample_bytree=0.85,
                      gamma=0.2,
                      subsample = 0.9,
#                              max_delta_step = 0.5,
                      reg_alpha=0.2,
                      reg_lambda=0.2,
                      seed=0,
                     )
xgb_model1 = xgb.XGBClassifier(objective="binary:logistic",
                      learning_rate =0.3,
                      scale_pos_weight=1,
                      min_child_weight=6,
                      max_depth=22,
                      n_estimators=110,
                      colsample_bytree=0.85,
                      gamma=0.2,
                      subsample = 0.9,
#                              max_delta_step = 0.5,
                      reg_alpha=0.2,
                      reg_lambda=0.6,
                      seed=0,
                     )
xgb_model2 = xgb.XGBClassifier(objective="binary:logistic",
                      learning_rate =0.3,
                      scale_pos_weight=1,
                      min_child_weight=6,
                      max_depth=18,
                      n_estimators=110,
                      colsample_bytree=0.85,
                      gamma=0.2,
                      subsample = 0.9,
#                              max_delta_step = 0.5,
                      reg_alpha=0.2,
                      reg_lambda=0.6,
                      seed=0,
                     )
xgb_model3 = xgb.XGBClassifier(objective="binary:logistic",
                      learning_rate =0.3,
                      scale_pos_weight=1,
                      min_child_weight=6,
                      max_depth=18,
                      n_estimators=110,
                      colsample_bytree=0.85,
                      gamma=0.2,
                      subsample = 0.9,
#                              max_delta_step = 0.5,
                      reg_alpha=0.2,
                      reg_lambda=0.2,
                      seed=0,
                     )
xgb_model4 = xgb.XGBClassifier(objective="binary:logistic",
                      learning_rate =0.1,
                      scale_pos_weight=1,
                      min_child_weight=6,
                      max_depth=18,
                      n_estimators=110,
                      colsample_bytree=0.85,
                      gamma=0.2,
                      subsample = 0.9,
#                              max_delta_step = 0.5,
                      reg_alpha=0.2,
                      reg_lambda=0.6,
                      seed=0,
                     )
clfs=[xgb_model,xgb_model1,xgb_model2,xgb_model3,xgb_model4]

print "Creating train and test sets for blending."

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    print j, clf
    dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        print "Fold", i
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:,1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
#print "Blending."
clf = LogisticRegression(C=0.2)
clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:,1]
#fn_WriteSubmission("blending_5xgb_with_1LR_C0.2 _ all way.csv",y_submission,y_id)
preds = pd.DataFrame({"Id":ID,"ACTION":y_submission})
preds = preds.set_index('Id')
preds.to_csv('final.csv')


#


#Blending weak classifiers on New Data to ensemble with previous file
np.random.seed(0) # seed to shuffle the train set

n_folds = 10
verbose = True
shuffle = False

data_trn = pd.read_csv('new_train.csv', index_col=False)
data_tst = pd.read_csv('new_test.csv', index_col=False)

target = ['ACTION']
feature_columns_to_use = [column for column in data_trn.columns if column not in target]

X = data_trn.as_matrix(feature_columns_to_use)
y = data_trn.as_matrix(['ACTION']).ravel()

X_submission = data_tst.as_matrix(feature_columns_to_use)
y_id = data_tst.as_matrix(['id']).ravel()

#X, y, X_submission = load_data.load()

if shuffle:
    idx = np.random.permutation(y.size)
    X = X[idx]
    y = y[idx]

skf = list(StratifiedKFold(y, n_folds))

clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

print "Creating train and test sets for blending."

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    print j, clf
    dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        print "Fold", i
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:,1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

print
print "Blending."
clf = LogisticRegression()
clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:,1]

print "Linear stretch of predictions to [0,1]"
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
y_id=range(1,58922)

def fn_WriteSubmission(_file_name, _y_predict, _y_id):
    # Write Submission
    submission = pd.DataFrame({"ACTION": _y_predict}, index=_y_id)
    submission.index.name = 'Id'

    submission.to_csv(_file_name)
#Call
fn_WriteSubmission("blending.csv",y_submission,y_id)


#


#Ensembling final.csv and blending.csv
loc_outfile = ‘hw3p3.csv'
def kaggle_bag(array, loc_outfile, weights=[0.8,0.2],method="average"):
  if method == "average":
    scores = defaultdict(float)
  with open(loc_outfile,"wb") as outfile:
    for i, glob_file in enumerate(array):
      print "parsing:", i,glob_file
      # sort glob_file by first column, ignoring the first line
      lines = open(glob_file).readlines()
      lines = [lines[0]] + sorted(lines[1:])
      for e, line in enumerate( lines ):
        if i == 0 and e == 0:
          outfile.write(line)
        if e > 0:
          row = line.strip().split(",")
          scores[(e,row[0])] += float(row[1])*weights[i]
    for j,k in sorted(scores):
      outfile.write("%s,%f\n"%(k,(scores[(j,k)])))
    print("wrote to %s"%loc_outfile)
array=[ os.getcwd()+'/final.csv', os.getcwd()+'/blending.csv']

kaggle_bag(array, loc_outfile)