import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')



# Feature Engineering Func
def trans_issueDate(issueDate):
    year, month, day = issueDate.split('-')
    return int(year)*12 + int(month)

def get_issueDate_day(issueDate):
    year, month, day = issueDate.split('-')
    return int(day)

def trans_earliesCreditLine(earliesCreditLine):
    month_dict = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, \
                  "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
    month, year = earliesCreditLine.split('-')
    month = month_dict[month]
    return int(year)*12 + month

def trans_employmentLength_num(employmentLength):
    if employmentLength=='10+ years':
        return 15
    elif employmentLength=='< 1 year':
        return 0
    else:
        return str(employmentLength)[:2]

employmentLength_dict = {'1 year':1,'10+ years':10,'2 years':2,'3 years':3,'4 years':4,
                         '5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9,'< 1 year':0}


cate_features = [
    'term', 
    'grade', 
    'subGrade', 
    'employmentTitle', 
    'employmentLength', 
    'homeOwnership',
    'verificationStatus', 
    'purpose', 
    'delinquency_2years', 
    'earliesCreditLine', 
    'postCode', 
    'regionCode', 
    'title', 
    'issueDate', 
    
    # bins_10
    'loanAmnt_bin', 
    'annualIncome_bin', 
    
    # bins_100
    'interestRate_bin', 
    'dti_bin', 
    'installment_bin', 
    'revolBal_bin',
    'revolUtil_bin'
    ]

def gen_new_feats(train, test):
    
    train['earliesCreditLine'] = train['earliesCreditLine'].apply(lambda x: trans_earliesCreditLine(x))
    test['earliesCreditLine'] = test['earliesCreditLine'].apply(lambda x: trans_issueDate(x))
    
    # Step 1: concat train & test -> data
    data = pd.concat([train, test])

    # Step 2.1 : Feature Engineering Part 1
    print('LabelEncoder...')
    encoder = LabelEncoder()
    data['grade'] = encoder.fit_transform(data['grade'])
    data['subGrade'] = encoder.fit_transform(data['subGrade'])
    data['postCode'] = encoder.fit_transform(data['postCode'])
    data['employmentTitle'] = encoder.fit_transform(data['employmentTitle'])
        
    print('generate new features...')
    # data['employmentLength'] = data['employmentLength'].apply(lambda x: trans_employmentLength_num(x))
    data['employmentLength'] = data['employmentLength'].apply(lambda x: x if x not in employmentLength_dict else employmentLength_dict[x])  
    data['issueDate_Day'] = data['issueDate'].apply(lambda x: get_issueDate_day(x))
    data['issueDate'] = data['issueDate'].apply(lambda x: trans_issueDate(x))
    data['date_Diff'] = data['issueDate'] - data['earliesCreditLine']  # 本次贷款距离上次的时间
    data['debt'] = data['dti'] * data['annualIncome']
    data['acc_ratio'] = data['openAcc'] / (data['openAcc'] + 0.1)
    data['revolBal_annualIncome_r'] = data['revolBal'] / (data['annualIncome'] + 0.1)
    data['revolTotal'] = 100*data['revolBal'] / (100 - data['revolUtil'])
    data['pubRec_openAcc_r'] = data['pubRec'] / (data['openAcc'] + 0.1)
    data['pubRec_totalAcc_r'] = data['pubRec'] / (data['totalAcc'] + 0.1)
    
    # step2.2: Binning
    print('Binning...')
    bin_nums = 10
    bin_labels = [i for i in range(bin_nums)]
    binning_features = ['loanAmnt', 'annualIncome']
    for f in binning_features:
        data['{}_bin'.format(f)] = pd.qcut(data[f], bin_nums, labels=bin_labels).astype(np.float64)
    bin_nums = 50
    bin_labels = [i for i in range(bin_nums)]
    binning_features = ['interestRate', 'dti', 'installment', 'revolBal','revolUtil']
    for f in binning_features:
        data['{}_bin'.format(f)] = pd.qcut(data[f], bin_nums, labels=bin_labels).astype(np.float64)
    
    for f in cate_features:
        data[f] = data[f].fillna(0).astype('int')
    
    return data[data['isDefault'].notnull()], data[data['isDefault'].isnull()]


def gen_target_encoding_feats(train, test, encode_cols, target_col, n_fold=10):
    '''生成target encoding特征'''
    # for training set - cv
    tg_feats = np.zeros((train.shape[0], len(encode_cols)))
    kfold = StratifiedKFold(n_splits=n_fold, random_state=2021, shuffle=True)
    for _, (train_index, val_index) in enumerate(kfold.split(train[encode_cols], train[target_col])):
        df_train, df_val = train.iloc[train_index], train.iloc[val_index]
        for idx, col in enumerate(encode_cols):
            target_mean_dict = df_train.groupby(col)[target_col].mean()
            df_val[f'{col}_mean_target'] = df_val[col].map(target_mean_dict)
            tg_feats[val_index, idx] = df_val[f'{col}_mean_target'].values

    for idx, encode_col in enumerate(encode_cols):
        train[f'{encode_col}_mean_target'] = tg_feats[:, idx]

    # for testing set
    for col in encode_cols:
        target_mean_dict = train.groupby(col)[target_col].mean()
        test[f'{col}_mean_target'] = test[col].map(target_mean_dict).astype(np.float64)

    return train, test

encoding_cate_features = [
    'term', 
    'grade', 
    'subGrade', 
    'employmentTitle', 
    'employmentLength', 
    'homeOwnership',
    'verificationStatus', 
    'purpose', 
    'delinquency_2years', 
    'earliesCreditLine', 
    'postCode', 
    'regionCode', 
    'title', 
    'issueDate', 
    
    # bins_10
    'loanAmnt_bin', 'annualIncome_bin', 
    
    # bins_100
    'interestRate_bin', 'dti_bin', 'installment_bin', 'revolBal_bin','revolUtil_bin'
    ]

TRAIN_FEAS = [
              #'id', 
              'loanAmnt', 
              'term', 
              'interestRate', 
              'installment', 
              'grade',
              'subGrade', 
              'employmentTitle', 
              'employmentLength', 
              'homeOwnership',
              'annualIncome', 
              'verificationStatus', 
              'issueDate', 
              'purpose', 
              'postCode', 
              'regionCode', 
              'dti', 
              'delinquency_2years',
              'ficoRangeLow', 
              # 'ficoRangeHigh', 
              'openAcc', 
              'pubRec',
              'pubRecBankruptcies', 
              'revolBal', 
              'revolUtil', 
              'totalAcc',
              'initialListStatus', 
              'applicationType', 
              'earliesCreditLine', 
              'title',
              'policyCode', 
              'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8',
              'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 
              'issueDate_Day', 'date_Diff','debt', 'acc_ratio', 
              'revolBal_annualIncome_r', 'revolTotal','pubRec_openAcc_r', 
              'pubRec_totalAcc_r', 'loanAmnt_bin','annualIncome_bin', 
              'interestRate_bin', 'dti_bin', 'installment_bin',
              'revolBal_bin', 'revolUtil_bin', 'term_mean_target',
              'grade_mean_target', 'subGrade_mean_target',
              'employmentTitle_mean_target', 'employmentLength_mean_target',
              'homeOwnership_mean_target', 'verificationStatus_mean_target',
              'purpose_mean_target', 'delinquency_2years_mean_target',
              'earliesCreditLine_mean_target', 'postCode_mean_target',
              'regionCode_mean_target', 'title_mean_target', 'issueDate_mean_target',
              'loanAmnt_bin_mean_target', 'annualIncome_bin_mean_target',
              'interestRate_bin_mean_target', 'dti_bin_mean_target',
              'installment_bin_mean_target', 'revolBal_bin_mean_target',
              'revolUtil_bin_mean_target'
            ]


cate_features=[
#     'term',
#  'grade',
#  'subGrade',
#  'employmentTitle',
#  'employmentLength',
#  'homeOwnership',
#  'verificationStatus',
#  'purpose',
#  'delinquency_2years',
#  'earliesCreditLine',
#  'postCode',
#  'regionCode',
#  'title',
#  'issueDate',
#  'loanAmnt_bin',
#  'annualIncome_bin',
#  'interestRate_bin',
#  'dti_bin',
#  'installment_bin',
#  'revolBal_bin',
#  'revolUtil_bin'
              ]

seed0=2021
lgb_param = {
    'objective': 'binary',  # 自定义
    'metric':'auc',
    'boosting_type': 'gbdt',
    
#     'max_bin':100,
#     'min_data_in_leaf':500,
    'learning_rate': 0.05,
    'subsample': 0.82,
    'subsample_freq': 1,
    'feature_fraction': 0.88,
    'lambda_l1': 6.1,
    'lambda_l2': 1.3,
    'max_depth':13,
    'min_child_weight': 18.5,
    'min_data_in_leaf': 97,
    'min_gain_to_split': 0.057,
    'num_leaves':24,
#     'categorical_column':[0],  # stock_id
    'seed':seed0,
    'feature_fraction_seed': seed0,
    'bagging_seed': seed0,
    'drop_seed': seed0,
    'data_random_seed': seed0,
    'n_jobs':-1,
    # 'device':'cuda',
    'verbose': -1}


cat_params = {
    'iterations': 50000, # 3000
    'depth':6,
    'l2_leaf_reg':5,
    'learning_rate': 0.02, # 0.05
    'loss_function':'CrossEntropy',
    'eval_metric': 'AUC',
    'task_type':'GPU',
    'random_seed': 2021,
    "early_stopping_rounds": 200,
    'verbose':100,
    # 'logging_level': 'Silent',
    'use_best_model': True,
    
}


def train_and_evaluate_lgb(train, test, params, split_seed):
    # Hyperparammeters (just basic)
    features = TRAIN_FEAS
    print('features num: ', len(features))
    print('cate features num: ', len(cate_features))
    y = train['isDefault']
    oof_predictions = np.zeros(train.shape[0])
    test_predictions = np.zeros(test.shape[0])
    kfold = KFold(n_splits = 5, random_state = split_seed, shuffle = True)

    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = train.iloc[trn_ind], train.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
        train_dataset = lgb.Dataset(x_train[features], y_train)
        val_dataset = lgb.Dataset(x_val[features], y_val)
        model = lgb.train(params = params,
                          num_boost_round = 10000,  # 1000
                          categorical_feature=cate_features,
                          train_set = train_dataset, 
                          valid_sets = [train_dataset, val_dataset], 
                          verbose_eval = 200,
                          early_stopping_rounds=150, # 50
                          )
        # Add predictions to the out of folds array
        oof_predictions[val_ind] = model.predict(x_val[features])
        # Predict the test set
        test_predictions += model.predict(test[features]) / 5
    score = roc_auc_score(y, oof_predictions)
    print(f'Our out of folds roc_auc is {score}')

    return test_predictions


def train_and_evaluate_cat(train, test, params, split_seed):
    # Hyperparammeters (just basic)
    
    y = train['isDefault']
    features = TRAIN_FEAS
    oof_predictions = np.zeros(train.shape[0])
    test_predictions = np.zeros(test.shape[0])

    nsplits = 5
    kfold = KFold(n_splits = nsplits, random_state = split_seed, shuffle = True)

    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
        
        model = CatBoostClassifier(**params)

        model.fit(x_train, 
                   y_train,  
                   eval_set=(x_val,y_val),
                #    eval_set=(test[:150000][features],test_a_tgt),
                   cat_features=cate_features,
                   use_best_model=True,
                   verbose=200
                  )
        
        # Add predictions to the out of folds array
        oof_predictions[val_ind] = model.predict_proba(x_val[features])[:,1]
        
        # Predict the test set
        # test_predictions += model.predict_proba(test[features])[:,1] / nsplits
        test_predictions += model.predict_proba(test[features])[:,1] / nsplits
    score = roc_auc_score(y, oof_predictions)
    print(f'Our out of folds roc_auc is {score}')

    return test_predictions


if __name__ == '__main__':
    # loading_data
    print('data loading...')
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test_a.csv')
    submit_sample = pd.read_csv('data/sample_submit.csv')
    print(train.shape, test.shape)

    # feature engineering
    print('feature engineering...')
    train, test = gen_new_feats(train, test)
    train, test = gen_target_encoding_feats(train, test, encoding_cate_features, target_col='isDefault', n_fold=10)
    print(train.shape, test.shape)

    # Train and Predict, Ensemble LGBM and Catboost
    print('Training and Predicting...')
    seed_list = [111,1024,2021]
    test_pred_lgb = np.zeros(test.shape[0])
    test_pred_cat = np.zeros(test.shape[0])

    for seed in tqdm(seed_list, total=len(seed_list)):
        test_pred_lgb += train_and_evaluate_lgb(train, test, lgb_param, split_seed=seed)/len(seed_list)
        test_pred_cat += train_and_evaluate_cat(train, test, cat_params, split_seed=seed)/len(seed_list)

    test_pred = 0.60 * test_pred_lgb + 0.40 * test_pred_cat

    # result
    submit_sample['isDefault'] = test_pred
    submit_sample.to_csv('DataBears THU Team 1 Outputs.csv',index=0)
    print('Complete!')

    # pd.Series(test_pred_cat).to_csv('data/cat_pred.csv',index=0)

