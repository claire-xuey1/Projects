# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import statsmodels.api as st
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import patsy
import graphviz #optional –needed to render a tree model into graph

os.chdir("/Users/xiangyifan/Desktop/SU/MSBA/19SQ/BUAN 5310 Statitstical Learning/group project/Gitgroupproject")
#airline = pd.read_csv("Airline_ANOVA.csv", sep = ',', header = 0)
#airline = pd.read_csv("Airline_forward.csv", sep = ',', header = 0)
#airline = pd.read_csv("Airline_backward.csv", sep = ',', header = 0)
data = pd.read_csv("full.csv", sep = ',', header = 0)


#airport

## variable selected
y_airport = data['Airport']
X_airport = data[['Airline', 'Nationality','FrequentFlightDestination','Destination','DepartureTime','Occupation','NoTransport','Airfare']]    

## Decision Tree           
X = pd.get_dummies(X_airport,drop_first=True,prefix_sep='_')
X_train, X_test, y_train, y_test = train_test_split(X, y_airport, test_size=0.30,random_state=109)
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')
clf = clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)   ## predict train set
y_pred_test = clf.predict(X_test)     ## predict test set
print(metrics.confusion_matrix(y_test, y_pred_test))
print("Decision Tree Train Accuracy for airport:",metrics.accuracy_score(y_train, y_pred_train))
print("Decision Tree Test Accuracy for airport:",metrics.accuracy_score(y_test, y_pred_test))

# export estimated tree into dot graphic file
dot_data = tree.export_graphviz(clf, out_file='Dtree_airport.dot',filled=True,rounded = True, feature_names=X.columns)

## MNLogit Model

interaction = "Airport ~  Nationality + Airline + FrequentFlightDestination + Destination +DepartureTime + Occupation + NoTransport + Airfare + Airfare : Destination"

y,XX = patsy.dmatrices(interaction, data, return_type = "dataframe")

X_train, X_test, y_train, y_test = train_test_split(XX, y_airport, test_size=0.30,random_state=109)
num_col_names = ['NoTransport','Airfare','Airfare:Destination[T.Japan]','Airfare:Destination[T.Other]','Airfare:Destination[T.SouthEast Asia]']   ## scale only numeric variable
scaler = StandardScaler().fit(X_train[num_col_names].values)
X_train[num_col_names] = scaler.transform(X_train[num_col_names].values)
X_test[num_col_names] = scaler.transform(X_test[num_col_names].values)
AirportLogit = st.MNLogit(y_train, X_train).fit_regularized()

print(AirportLogit.summary())
y_prob_train = AirportLogit.predict(X_train)
y_pred_train = y_airport.astype('category').cat.categories[y_prob_train.idxmax(axis=1)]
y_prob_test = AirportLogit.predict(X_test)
y_pred_test = y_airport.astype('category').cat.categories[y_prob_test.idxmax(axis=1)]

print(metrics.confusion_matrix(y_test, y_pred_test))
print("MNLogit Train Accuracy for airport:",metrics.accuracy_score(y_train, y_pred_train))
print("MNLogit Test Accuracy for airport:",metrics.accuracy_score(y_test, y_pred_test))

## SVM 
X_train=X_train.drop(['Airfare:Destination[T.Japan]','Airfare:Destination[T.Other]','Airfare:Destination[T.SouthEast Asia]'], axis=1)
X_test=X_test.drop(['Airfare:Destination[T.Japan]','Airfare:Destination[T.Other]','Airfare:Destination[T.SouthEast Asia]'], axis=1)
svclassifier = SVC(C = 1.8, kernel='rbf')    
svclassifier.fit(X_train, y_train)  
y_pred_train = svclassifier.predict(X_train)    ## predict train set
y_pred_test = svclassifier.predict(X_test)  	## predict test set
print(metrics.confusion_matrix(y_test, y_pred_test))
print("RBF SVM Train Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("RBF SVM Test Accuracy:",metrics.accuracy_score(y_test, y_pred_test))


## Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(350,500), random_state=109,max_iter=5000)
mlp.fit(X_train, y_train)
y_pred_train = mlp.predict(X_train)    ## predict train set
y_pred_test = mlp.predict(X_test)  	## predict test set
print(metrics.confusion_matrix(y_test, y_pred_test))
print("Neural Network Test Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("Neural Network Test Accuracy:",metrics.accuracy_score(y_test, y_pred_test))





#airline

y_airline = data['Airline']

## Decision Tree
col_names = ['Destination', 'Airfare', 'DepartureTime', 'Airport',
       'FlyingCompanion', 'NoTransport', 'FrequentFlightDestination',
       'TripDuration', 'NoTripsLastYear']     ## variable selected
airline_tree= data[col_names]
#airline_tree = airline.drop(['Airline'], axis=1)  
                        
X = pd.get_dummies(airline_tree,drop_first=True,prefix_sep='_')
X_train, X_test, y_train, y_test = train_test_split(X, y_airline, test_size=0.30,random_state=109)
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')
clf = clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)   ## predict train set
y_pred_test = clf.predict(X_test)     ## predict test set
print(metrics.confusion_matrix(y_test, y_pred_test))
print("Decision Tree Train Accuracy for airline:",metrics.accuracy_score(y_train, y_pred_train))
print("Decision Tree Test Accuracy for airline:",metrics.accuracy_score(y_test, y_pred_test))


# export estimated tree into dot graphic file
dot_data = tree.export_graphviz(clf, out_file='Dtree_airline.dot',filled=True,rounded = True, feature_names=X.columns)


## MNLogit Model

interaction2 = "Airline ~  Age + FlyingCompanion + FlyingCompanion : Airfare + Airfare + TripDuration + Airport + Destination +DepartureTime + Occupation + Income + Airfare : Destination"

y,XX2 = patsy.dmatrices(interaction2, data, return_type = "dataframe")


X_train, X_test, y_train, y_test = train_test_split(XX2, y_airline, test_size=0.30,random_state=109)
num_col_names = ['Age','FlyingCompanion','TripDuration','Airfare','Airfare:Destination[T.Japan]','Airfare:Destination[T.Other]','Airfare:Destination[T.SouthEast Asia]']   ## scale only numeric variable
scaler = StandardScaler().fit(X_train[num_col_names].values)
X_train[num_col_names] = scaler.transform(X_train[num_col_names].values)
X_test[num_col_names] = scaler.transform(X_test[num_col_names].values)
AirlineLogit = st.MNLogit(y_train, X_train).fit()

print(AirlineLogit.summary())
y_prob_train = AirlineLogit.predict(X_train)
y_pred_train = y_airline.astype('category').cat.categories[y_prob_train.idxmax(axis=1)]
y_prob_test = AirlineLogit.predict(X_test)
y_pred_test = y_airline.astype('category').cat.categories[y_prob_test.idxmax(axis=1)]

print(metrics.confusion_matrix(y_test, y_pred_test))
print("MNLogit Train Accuracy for airline:",metrics.accuracy_score(y_train, y_pred_train))
print("MNLogit Test Accuracy for airline:",metrics.accuracy_score(y_test, y_pred_test))



## SVM
col_names = ['Destination', 'Airfare', 'DepartureTime', 'Airport',
       'FlyingCompanion', 'NoTransport', 'FrequentFlightDestination',
       'TripDuration', 'NoTripsLastYear']     ## variable selected
tables = "Airline ~  Destination + Airfare+ DepartureTime +  NoTripsLastYear +  Airport +TripDuration + FrequentFlightDestination + NoTransport + FlyingCompanion"     

y,SVM = patsy.dmatrices(tables, data, return_type = "dataframe")

X_train, X_test, y_train, y_test = train_test_split(SVM, y_airline, test_size=0.30,random_state=109)
num_col_names = ['FlyingCompanion','TripDuration','Airfare','NoTripsLastYear','NoTransport']   ## scale only numeric variable
scaler = StandardScaler().fit(X_train[num_col_names].values)
X_train[num_col_names] = scaler.transform(X_train[num_col_names].values)
X_test[num_col_names] = scaler.transform(X_test[num_col_names].values)
svclassifier = SVC(kernel='linear')    	## Linear SVM
svclassifier.fit(X_train, y_train)
y_pred_train = svclassifier.predict(X_train)    ## predict train set
y_pred_test = svclassifier.predict(X_test)  	## predict test set
print(metrics.confusion_matrix(y_test, y_pred_test))
print("Linear SVM Train Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("Linear SVM Test Accuracy:",metrics.accuracy_score(y_test, y_pred_test))


svclassifier = SVC(kernel='poly', degree=3)    	## cubic polynomial SVM
svclassifier.fit(X_train, y_train)
y_pred_train = svclassifier.predict(X_train)
y_pred_test = svclassifier.predict(X_test)  	## predict test set
print(metrics.confusion_matrix(y_test, y_pred_test))
print("cubic polynomial SVM Train Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("cubic polynomial SVM Test Accuracy:",metrics.accuracy_score(y_test, y_pred_test))

svclassifier = SVC(kernel='rbf')    	## RBF SVM
svclassifier.fit(X_train, y_train)
y_pred_train = svclassifier.predict(X_train)    ## predict train set
y_pred_test = svclassifier.predict(X_test)  	## predict test set
print(metrics.confusion_matrix(y_test, y_pred_test))
print("RBF SVM Test Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("RBF SVM Test Accuracy:",metrics.accuracy_score(y_test, y_pred_test))


## Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), random_state=109,max_iter=5000)
mlp.fit(X_train, y_train)
y_pred_train = mlp.predict(X_train)    ## predict train set
y_pred_test = mlp.predict(X_test)  	## predict test set
print(metrics.confusion_matrix(y_test, y_pred_test))
print("Neural Network Test Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("Neural Network Test Accuracy:",metrics.accuracy_score(y_test, y_pred_test))
