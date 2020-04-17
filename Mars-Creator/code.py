# --------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Code starts here
df = pd.read_csv(filepath_or_buffer=path, compression='zip')

print(df.head())

X = df.drop(columns = ['attr1089'])

y = df['attr1089']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=4)

scaler= MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_test[45,5])

# Code ends here


# --------------
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
#Initialize the logistic regression
lr = LogisticRegression()
#Fit the model
lr.fit(X_train, y_train)
#Store the predicted values of test data
y_pred = lr.predict(X_test)
#roc score
roc_score = roc_auc_score(y_pred, y_test)


# --------------
from sklearn.tree import DecisionTreeClassifier
#Initialise the Decision Tree
dt = DecisionTreeClassifier(random_state=4)
#Fit the model
dt.fit(X_train, y_train)
#Predict the model
y_pred = dt.predict(X_test)
#Check the roc score
roc_score = roc_auc_score(y_pred, y_test)


# --------------
from sklearn.ensemble import RandomForestClassifier


# Code strats here
#intialise the model
rfc = RandomForestClassifier(random_state=4)
#Fit the model
rfc.fit(X_train, y_train)
# Predcit the model
y_pred = rfc.predict(X_test)
#roc score
roc_score = roc_auc_score(y_pred, y_test)
# Code ends here


# --------------



# Import Bagging Classifier
from sklearn.ensemble import BaggingClassifier
#Initialise the model
bagging_clf = BaggingClassifier(DecisionTreeClassifier(), random_state=0,n_estimators=100,max_samples=100) 

#fit the model
bagging_clf.fit(X_train, y_train)
#predict the model
y_pred = bagging_clf.predict(X_test)
#roc score of the model

# Code starts here
print(bagging_clf.score(X_test, y_test))
score_bagging = roc_auc_score(y_test, y_pred)
print(score_bagging)
# Code ends here


# --------------
# Import libraries
from sklearn.ensemble import VotingClassifier

# Various models
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier(random_state=4)
clf_3 = RandomForestClassifier(random_state=4)

model_list = [('lr',clf_1),('DT',clf_2),('RF',clf_3)]


# Code starts here
#Initialise the model
voting_clf_hard = VotingClassifier(estimators = model_list, voting="hard")
#fit the model
voting_clf_hard.fit(X_train, y_train)
#Score of the model
hard_voting_score = voting_clf_hard.score(X_test, y_test)
print(hard_voting_score)


# Code ends here


