import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor as GBT
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('./data/House_Prices/train.csv')

data.head()

'''
  Data obtained from https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv.
  The data contains 1460 datapoints of property prices along with 80 features describing the qualities.
  Many values are missing from data so a simple null-imputation method was used to fill in numeric columns with zeros.

  We will look at 3 different models to approach this problem, namely:
  1. Ridge regression
  - Since there are many features presented, we cannot use the regular linear regression to solve this regression problem due to the curse of dimensionality.
  - Ridge regression imposes a penality on the overall magnitude of the coefficients given to each feature, causing an overall shrinkage of the model.
  - As a results, the effect of insignificant features will be diminished and the model performs better than the regular linear regression mode.
  2. Lasso
  - The lasso works similarly to the ridge regression, but penalises the coefficients more heavily. 
  - Insignificant features will have their coefficients shrunk down to zero (unlike ridge regression where these coefficients may be very small but non-zero).
  - This effectively performs parameter selection in the process and cause the resulting model to be more accurate.
  3. Gradient Boosted Regression (tree-based)
  - This model works by fitting simple tree model iteratively to improve the prediction performance.
  - In each iteration of fitting, a new model is added to fix the error created by the existing set of models, thus causing the prediction performance to improve.
  - This mechanism works with any models; in this example we choose the decision tree to be our base model to contrast its performance with the linear ridge regression and lasso.
'''

##### Dataset
X = data.drop('SalePrice',axis=1)
y = data['SalePrice']

# perform train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

##### Ridge regression
# then standardise each set ** AFTER ** the split to prevent bias
# note that we do not need to standardise the target (y)
X_train = (X_train-X_train.mean())/X_train.std()
X_test = (X_test-X_test.mean())/X_test.std()

X_train_filled = X_train.fillna(0)
X_test_filled = X_test.fillna(0)

ridge = Ridge()
param_grid = {
    'alpha': [2,1.51,0.5,0.1,0.01]
}

cv = GridSearchCV(
    ridge,
    param_grid,
    verbose=1
)
cv.fit(X_train_filled,y_train)

ridge_score = cv.best_estimator_.score(X_test_filled,y_test)
print(cv.best_params_)
print(f'Thest best prediction performance is {round(ridge_score,2)}')

features = pd.DataFrame({'feature':X_train.columns,'feature_importance':cv.best_estimator_.coef_})
features.sort_values('feature_importance',ascending=False).reset_index(drop=True).head(10)
print(features)

##### Lasso
X_train_filled = X_train.fillna(0)
X_test_filled = X_test.fillna(0)

lasso = Lasso()
param_grid = {
    'alpha': [2,1.51,0.5,0.1,0.01]
}

cv = GridSearchCV(
    lasso,
    param_grid,
    verbose=1
)
cv.fit(X_train_filled,y_train)

lasso_score = cv.best_estimator_.score(X_test_filled,y_test)
print(cv.best_params_)
print(f'Thest best prediction performance is {round(lasso_score,2)}')

features = pd.DataFrame({'feature':X_train.columns,'feature_importance':cv.best_estimator_.coef_})
features.sort_values('feature_importance',ascending=False).reset_index(drop=True).head(10)
print(features)

##### Gradient Boosted Regressor
gbt = GBT()
param_grid = {
    'learning_rate': [0.1,0.5,1],
    'n_estimators': [100,200,300]
}

cv = GridSearchCV(
    gbt,
    param_grid,
    verbose=1
)
cv.fit(X_train_filled,y_train)

gbt_score = cv.best_estimator_.score(X_test_filled,y_test)
print(cv.best_params_)
print(f'Thest best prediction performance is {round(gbt_score,2)}')

best_gbt = cv.best_estimator_
features = pd.DataFrame({'feature':X_train.columns,'feature_importance':best_gbt.feature_importances_})
features.sort_values('feature_importance',ascending=False).reset_index(drop=True).head(10)
print(features)

############################################################

print('-'*30)
print(f'Ridge Regression R^2 is {round(ridge_score,2)}')
print(f'Lasso R^2 Tree is {round(lasso_score,2)}')
print(f'Gradient Boosted Regressor R^2 is {round(gbt_score,2)}')

# output the best prediction
data['SalePrice_pred'] = best_gbt.predict(
    pd.concat([X_train,X_test])
)
data.to_csv('./output/regression/gbt-saleprice_pred.csv')