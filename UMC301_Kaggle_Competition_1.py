import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# List of categorical features to group by
categorical_features = ['key', 'audio_mode', 'time_signature']

# List of numerical features to aggregate
numerical_features = ['danceability', 'energy', 'loudness', 'acousticness', 'tempo', 'audio_valence']

for cat_col in categorical_features:
    for num_col in numerical_features:
        agg_stats = train_df.groupby(cat_col)[num_col].agg(['mean', 'std']).reset_index()
        agg_stats.columns = [cat_col, f'{num_col}_by_{cat_col}_mean', f'{num_col}_by_{cat_col}_std']

        train_df = pd.merge(train_df, agg_stats, on=cat_col, how='left')
        test_df = pd.merge(test_df, agg_stats, on=cat_col, how='left')

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

X = train_df.drop(columns=['id', 'song_popularity'])
y = train_df['song_popularity']
X_test = test_df.drop(columns=['id'])

train_cols = X.columns
test_cols = X_test.columns
shared_cols = list(set(train_cols) & set(test_cols))
X = X[shared_cols]
X_test = X_test[shared_cols]


imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.20],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.4, 0.7, 0.8],
    'n_estimators': [500, 1000, 1500],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.001, 0.005], 
    'reg_lambda': [0.1, 1, 5]
}

xgb_for_search = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=57,
    n_jobs=-1
)
random_search = RandomizedSearchCV(
    estimator=xgb_for_search,
    param_distributions=param_grid,
    n_iter=50,
    scoring='roc_auc',
    cv=3,
    verbose=2,
    random_state=57,
    n_jobs=-1
)
random_search.fit(X, y)

print(f"Best AUC score found: {random_search.best_score_:.5f}")
print("Best parameters found: ", random_search.best_params_)

best_model = random_search.best_estimator_
test_predictions = best_model.predict_proba(X_test)[:, 1]

submission_df = pd.DataFrame({'id': test_df['id'], 'song_popularity': test_predictions})
submission_df.to_csv('submission_final.csv', index=False)
print(submission_df.head())
