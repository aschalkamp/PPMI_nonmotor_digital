import numpy as np
import pandas as pd

import os

import seaborn as sns
import pylab as plt
import plots

from sklearn import preprocessing, metrics,model_selection,linear_model,compose
from sklearn.pipeline import Pipeline

def run_regression(data,features,covs,target='updrs_ii',save='/scratch/c.c21013066/data/ppmi/analyses/predictclinical'):
    # Define a pipeline to search for the best classifier regularization.
    # Define a Standard Scaler to normalize inputs
    cvouter = model_selection.KFold(n_splits=5,random_state=12,shuffle=True)
    preds = np.hstack([features,covs])
    coefs_df = pd.DataFrame(columns=['coef'],index=pd.MultiIndex.from_product([np.arange(5),np.hstack([preds,'intercept'])],names=['cv','predictor']))
    test_scores_df = pd.DataFrame(index=np.arange(5),columns=['r2'])
    fitted_params_df = pd.DataFrame(index=np.arange(5),columns=['alpha','l1_ratio'])
    
    for cvo,(train_index, test_index) in enumerate(cvouter.split(data[preds])):
        X_train, X_test = data.loc[train_index,preds], data.loc[test_index,preds]
        y_train, y_test = data.loc[train_index,target], data.loc[test_index,target]

        cols_to_scale = list(range(0, X_train.shape[1]-2))

        # Create a preprocessor
        scaler = compose.ColumnTransformer(
            transformers=[
                ('scale', preprocessing.StandardScaler(), cols_to_scale)],
            remainder='passthrough')
        targetscaler = preprocessing.StandardScaler()
        cv = model_selection.KFold(n_splits=5,random_state=123,shuffle=True)
        # set the tolerance to a large value to make the example faster
        linear = linear_model.ElasticNet(max_iter=10000, tol=0.1,random_state=4,fit_intercept=True)
        pipe = Pipeline(steps=[("scaler", scaler), ("elasticnet", linear)])
        param_grid = {
            'regressor__elasticnet__alpha': np.logspace(-2, 2, 5), 
            'regressor__elasticnet__l1_ratio': np.arange(0.1, 1, 0.1)
        }

        regr = compose.TransformedTargetRegressor(regressor=pipe, transformer=targetscaler)
        search = model_selection.GridSearchCV(regr, param_grid, n_jobs=-1,cv=cv,scoring='r2')
        search.fit(X_train[preds],y_train)

        # For each number of components, find the best classifier results
        coefs_df.loc[(cvo,preds),:] = search.best_estimator_.regressor_.named_steps['elasticnet'].coef_.T
        coefs_df.loc[(cvo,'intercept'),:] = search.best_estimator_.regressor_.named_steps['elasticnet'].intercept_
        test_scores_df.loc[cvo,'r2'] = search.best_score_
        fitted_params_df.loc[cvo,['alpha']] = search.best_params_['regressor__elasticnet__alpha']
        fitted_params_df.loc[cvo,['l1_ratio']] = search.best_params_['regressor__elasticnet__l1_ratio']

 
    if save:
        if not os.path.exists(save):
            os.makedirs(save)
        # save the dataframes to the directory as CSV files
        coefs_df.reset_index().to_csv(os.path.join(save, 'coefs.csv'))
        test_scores_df.to_csv(os.path.join(save, 'test_scores.csv'))
        fitted_params_df.to_csv(os.path.join(save, 'fitted_params.csv'))
        try:
            plots.plot_coefs(coefs_df,save=save)
        except:
            print('failed plotting for ',target)
    
    return coefs_df, search