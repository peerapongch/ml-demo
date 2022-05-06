import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
import warnings
warnings.filterwarnings("ignore")


##### Dataset
'''
  Dataset obtained from https://www.kaggle.com/datasets/muratkokludataset/date-fruit-datasets
  The dataset was created for classification of date fruit type to replace human effort, containing 898 datapoints of 34 features about a date fruit and 1 class label of type.
  There are 7 date fruit types in total, so this problem requires a multiclass classifcation model to solve.

  Here we will look at 3 types of multiclass classification model, namely:
  1. K-NN:
  - This model learn to label an unseen fruit based on how closely related its features are to seen fruits.
  - The method is model free and will work with any numeric or one-hot-encoded features.
  - In this example we will be using the Euclidean distance to measure the similarity, although users can tweak the distance measures when appropriate (e.g. Mahalanobis distance).
  2. Decision Tree:
  - This model will construct a tree node-wise where each node is a cut-point in the feature space of a selected feature.
  - The cut-point is calculated such that the child node will contain less impure data i.e. maximising number of datapoints of the same class.
  - In this example, we will look at two strategies of calculating the cut-point i.e. using gini vs. entropy score of a feature.
  - The best part about this model is that we get a nice table of feature importance and a model with tree structure that make decision making traceable i.e. explainable AI.
  3. Random Forest
  - This model constructs hundreds of Decision Trees with different conditions i.e. some features are available to certain tree but not the other.
  - This approach helps eliminate random errors by using 'wisdom of the crowd' for decision making.
  - This results in the best results out of the three proposed model, but the downside is less interpretability than the decision tree.

  To abide with the best practice, we don't just fit each model with any hyper-parameter. We will use grid search 5-fold cross validation to fine-tune model with the best parameter.
  The prediction score of each model is measured by the accuracy, displayed at the end of each section. 
'''
data = pd.read_excel('./data/Date_Fruit_Datasets/Date_Fruit_Datasets.xlsx')

##### prepare dataset
X = data.drop('Class',axis=1)
y = data['Class']

###### perform train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

###### then standardise each set ** AFTER ** the split to prevent bias
X_train = (X_train-X_train.mean())/X_train.std()
X_test = (X_test-X_test.mean())/X_test.std()

print('-'*30)
###### fitting knn
knn = KNN()
param_grid = {
    'n_neighbors': list(range(21))
}

cv = GridSearchCV(
    knn,
    param_grid,
    verbose=1
)
cv.fit(X_train,y_train)

knn_score = cv.best_estimator_.score(X_test,y_test)
print(cv.best_params_)
print(f'Thest best prediction accuracy is {round(knn_score*100,2)}%')

print('-'*30)
###### fitting decision trees
dt = DT()
param_grid = {
    'criterion': ['gini','entropy'],
    'max_depth': list(range(3,10))
}

cv = GridSearchCV(
    dt,
    param_grid,
    verbose=1
)
cv.fit(X_train,y_train)

dt_score = cv.best_estimator_.score(X_test,y_test)
print(cv.best_params_)
print(f'Thest best prediction accuracy is {round(dt_score*100,2)}%')

# explain random forest
best_dt = cv.best_estimator_
features = pd.DataFrame({'feature':X_train.columns,'feature_importance':best_dt.feature_importances_})
features.sort_values('feature_importance',ascending=False).reset_index(drop=True).head(10)
print(features)

print('-'*30)
# fitting random forest
rf = RF()
param_grid = {
    'n_estimators': [100,200,300],
    'max_depth': [None,1,3,5]
}

cv = GridSearchCV(
    rf,
    param_grid,
    verbose=1
)
cv.fit(X_train,y_train)

rf_score = cv.best_estimator_.score(X_test,y_test)
print(cv.best_params_)
print(f'Thest best prediction accuracy is {round(rf_score*100,2)}%')

best_rf = cv.best_estimator_
features = pd.DataFrame({'feature':X_train.columns,'feature_importance':best_rf.feature_importances_})
features.sort_values('feature_importance',ascending=False).reset_index(drop=True).head(10)
print(features)

print('-'*30)
print(f'K-Nearest Neighbors accuracy is {round(knn_score*100,2)}%')
print(f'Decision Tree accuracy is {round(dt_score*100,2)}%')
print(f'Random Forest accuracy is {round(rf_score*100,2)}%')

# output the best prediction
data['Class_pred'] = best_rf.predict(
    pd.concat([X_train,X_test])
)
data.to_csv('./output/classification/rf-class_pred.csv')