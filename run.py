

import polars as pl
import lightgbm as lgb

from pathlib import Path
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
from preproc import preprocessor

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 10,  
    "learning_rate": 0.05,
    "n_estimators": 2000,  
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,
    "verbose": -1,
    "random_state": 42,
    "reg_alpha": 0.1,
    "reg_lambda": 10,
    "extra_trees":True,
    'num_leaves':64,
    "device": device, 
    "verbose": -1,
}

fitted_models_cat = []
fitted_models_lgb = []
cv_scores_cat = []
cv_scores_lgb = []


def run(dir_path):
	train_df = preprocessor(dir_path)
	train_num_df = train_df.select(pl.exclude(pl.Categorical)).pipe(select_num_features)
	train_cat_df = train_df.select(pl.Categorical)
	train_data = pl.concat([train_num_df, train_cat_df], how='horizontal')
	cat_cols = train_cat_df.columns
	y = train_data.select(pl.col('target')).collect().to_pandas()
	weeks = train_data.select(pl.col('WEEK_NUM')).collect().to_pandas()
	train_data = train_data.drop(['case_id', 'target', 'WEEK_NUM']).collect().to_pandas()
	train_data[cat_cols] = train_data[cat_cols].astypr(str)
	cv = StratifiedGroupKFold(n_splits=5, shuffle=False)


	for train_idx, valid_idx in cv.split(train_data, y, groups=weeks):
	    X_train, y_train = train_data.iloc[train_idx], train_data.iloc[train_idx]
	    X_valid, y_valid = train_data.iloc[valid_idx], train_data.iloc[valid_idx]
	    train_pool = Pool(X_train, y_train,cat_features=cat_cols)
	    val_pool = Pool(X_valid, y_valid,cat_features=cat_cols)
	    
	    clf = CatBoostClassifier(
	    eval_metric='AUC',
	    task_type='GPU',
	    learning_rate=0.03,
	    iterations=n_est)
	    random_seed=3107
	    clf.fit(train_pool, eval_set=val_pool,verbose=300)
	    fitted_models_cat.append(clf)
	    
	    y_pred_valid = clf.predict_proba(X_valid)[:,1]
	    auc_score = roc_auc_score(y_valid, y_pred_valid)
	    cv_scores_cat.append(auc_score)

	    train_data[cat_cols] = train_data[cat_cols].astypr("categorical")

	    model = lgb.LGBMClassifier(**params)
	    model.fit(
	        X_train, y_train,
	        eval_set = [(X_valid, y_valid)],
	        callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)] )
	    
	    fitted_models_lgb.append(model)
	    y_pred_valid = model.predict_proba(X_valid)[:,1]
	    auc_score = roc_auc_score(y_valid, y_pred_valid)
	    cv_scores_lgb.append(auc_score)
	    
    
	print("CV AUC scores: ", cv_scores_cat)
	print("Maximum CV AUC score: ", max(cv_scores_cat))


	print("CV AUC scores: ", cv_scores_lgb)
	print("Maximum CV AUC score: ", max(cv_scores_lgb))



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='run the trainig models.')
	parser.add_argument('--dir_path', metavar='path', required=True, help='the path to data files')
	args = parser.parse_args()
	run(args.dir_path)