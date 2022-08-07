import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def pre_processing(file = None):
    import pandas as pd
    
    if file is None: 
        raise Exception('a file path or pandas dataframe must be given')
        
    if type(file)==str:
        try:
            dataset = pd.read_csv(file,sep='\t', low_memory = False)
        except: 
            raise Exception('file path {} not found'.format(file))
            
    elif type(file)!=pd.core.frame.DataFrame:
        raise Exception('a file path or pandas dataframe must be given'.format(file))
        
    if type(file)==pd.core.frame.DataFrame:
        dataset = file
            
    categorical_elements = ['term',
                            'initial_list_status',
                            'application_type',
                            'grade',
                            'home_ownership',
                            'verification_status',
                            'purpose']
    
    categorical_features = ['term_ 60 months', 'initial_list_status_w', 'application_type_JOINT',
                            'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G',
                            'home_ownership_MORTGAGE', 'home_ownership_NONE',
                            'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
                            'verification_status_Source Verified', 'verification_status_Verified',
                            'purpose_credit_card', 'purpose_debt_consolidation',
                            'purpose_educational', 'purpose_home_improvement', 'purpose_house',
                            'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
                            'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
                            'purpose_vacation', 'purpose_wedding']
    
    date_columns=['last_pymnt_d',
                  'last_credit_pull_d',
                  'issue_d',
                  'earliest_cr_line']
    
    num_features = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
                    'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
                    'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_bal',
                    'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt',
                    'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'last_pymnt_amnt', 'tot_coll_amt',
                    'tot_cur_bal', 'total_rev_hi_lim']
    
    derived_features = ['months_since_last_payment','months_since_last_credit_pull',
                        'months_since_earliest_cr_line',
                        'emp_length','late_fees_applied']

    # Clipping the values to 95% Quantile
    for column in num_features:
        upp = dataset[column].quantile(0.975)
        low = dataset[column].quantile(0.025)
        dataset[column] = dataset[column].clip(upper = upp, lower = low)
    
    categorical_dummies =  pd.get_dummies(dataset[categorical_elements], drop_first=True)
    
    for column in categorical_features:
        if column not in categorical_dummies.columns:
            categorical_dummies[column] = 0
            categorical_dummies[column] = categorical_dummies[column].astype('uint8')

    dataset = pd.concat([dataset, categorical_dummies], axis = 1)
    
     #Converting Date Columns to Number of Months since
    for column in date_columns:
            dataset[column]=pd.to_datetime(dataset[column])

    dataset['current_date'] = pd.to_datetime("2017-01-01")

    dataset['months_since_last_payment'] = ((dataset.current_date.dt.year  - dataset.last_pymnt_d.dt.year) * 12 +
                                         (dataset.current_date.dt.month - dataset.last_pymnt_d.dt.month))
    dataset['months_since_last_credit_pull'] = ((dataset.current_date.dt.year  - dataset.last_credit_pull_d.dt.year) * 12 +
                                         (dataset.current_date.dt.month - dataset.last_pymnt_d.dt.month))
    # dataset['months_since_issue'] = ((dataset.current_date.dt.year  - dataset.issue_d.dt.year) * 12 +
    #                                      (dataset.current_date.dt.month - dataset.last_pymnt_d.dt.month))
    dataset['months_since_earliest_cr_line'] = ((dataset.current_date.dt.year  - dataset.earliest_cr_line.dt.year) * 12 +
                                         (dataset.current_date.dt.month - dataset.last_pymnt_d.dt.month))
    emp_dict = {'< 1 year':0, '1 year':1, '2 years':2,
                '3 years':3, '4 years':4, '5 years':5,
                '6 years':6, '7 years':7, '8 years':8,
                '9 years':9, '10+ years':10}
    dataset['emp_length'] = dataset['emp_length'].map(emp_dict)
    
    # derived feature for to check if late fees were applied
    dataset['late_fees_applied']= np.where(dataset['total_rec_late_fee']>0, 1, 0)
    
    # filling in nulls - 0 for employment length, median for all others numerical and derived features.
    dataset['emp_length'] = dataset['emp_length'].fillna(0)
    
    for column in num_features + derived_features:
        dataset[column]=dataset[column].fillna(dataset[column].median())
    
    all_features = num_features+categorical_features+derived_features+['issue_d']
    
    return dataset[all_features+['default_ind']], all_features


def data_split(dataset, all_features, random_split = False ,undersample = False, oversample = False):

    if undersample & oversample:
        return print('please pick only one - over or under sample method')
    
    if random_split == True:
        X=dataset.drop('issue_d',axis=1)
        y=dataset[['default_ind']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify = y)
    else:
        X=dataset[all_features]
        y=dataset[['default_ind','issue_d']]

        X_train=X[X['issue_d']<='2015-05-31'].drop('issue_d',axis=1)
        X_test=X[X['issue_d']>'2015-05-31'].drop('issue_d',axis=1)

        y_train=y[y['issue_d']<='2015-05-31'].drop('issue_d',axis=1)
        y_test=y[y['issue_d']>'2015-05-31'].drop('issue_d',axis=1)
    
    if undersample == True:
        undersampler = RandomUnderSampler(sampling_strategy='majority')
        X_train,y_train = undersampler.fit_resample(X_train, y_train)
        
    if oversample == True:
        oversampler = SMOTE()
        undersampler = RandomUnderSampler(sampling_strategy=0.461556)
        X_train,y_train = undersampler.fit_resample(X_train, y_train)
        X_train,y_train = oversampler.fit_resample(X_train, y_train)

    return X_train, X_test, np.ravel(y_train), np.ravel(y_test)

