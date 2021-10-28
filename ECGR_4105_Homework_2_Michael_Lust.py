#!/usr/bin/env python
# coding: utf-8

# In[1234]:


#Homework Assignment 2
#Michael Lust
#Student ID: 801094861
#4105 Intro to Machine Learning
#October 28, 2021


# In[1235]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[1236]:


dataset = pd.read_csv("diabetes.csv")


# In[1237]:


dataset.head()


# In[1238]:


M = len(dataset)
M


# In[1239]:


dataset.head(20)


# In[1240]:


#Our Data set will consider Pregnacies through Age as Independent variables (X1-X8) and Outcome as Dependent (Y). 
X=dataset.iloc[:,[0,1,2,3,4,5,6,7]].values
Y=dataset.iloc[:,8].values
#see here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.
X[0:10] #Shows the Array


# In[1241]:


#Now we’ll split our Data set into Training Data and Test Data.
#Logistic model and Test data will be used to validate our model. We’ll use Sklearn.
from sklearn.model_selection import train_test_split

np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size =0.8, test_size = 0.2, random_state = 1)
X_train.shape


# In[1242]:


X_test.shape


# In[1243]:


Y_train.shape


# In[1244]:


Y_test.shape


# In[1245]:


#Now we’ll do feature scaling to scale our data between -1 and 1 using standardization.
#Here Scaling is important because there is a significant difference between explanatory variables.
from sklearn.preprocessing import StandardScaler #Assignment said to use Standardization only.
scalar_X = StandardScaler()
X_train = scalar_X.fit_transform(X_train) #New X_train is scaled
X_test = scalar_X.transform(X_test) #New X_test is scaled


# In[1246]:


#Problem 1


# In[1247]:


#Import LogisticRegression from sklearn.linear_model
#Make an instance classifier of the object LogisticRegression and give random_state = 0 
from sklearn.linear_model import LogisticRegression
classifier_L = LogisticRegression(random_state=0)
classifier_L.fit(X_train,Y_train)


# In[1248]:


Y_pred = classifier_L.predict(X_test)


# In[1249]:


Y_pred[0:9]


# In[1250]:


#Using Confusion matrix representing binary classifiers so we can get accuracy of our model.
from sklearn.metrics import confusion_matrix 
cnf_matrix = confusion_matrix(Y_test,Y_pred)
cnf_matrix


# In[1251]:


#We are evaluating the model using model evaluation metrics for accuracy, precision, and recall.
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))
print("Recall:",metrics.recall_score(Y_test,Y_pred))


# In[1252]:


#Let's visualize the results of the model in the form of a confusion matrix using matplot
#Here, you will visualize the confusion matrix using Heatmap.
import seaborn as sns 
class_names = [0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix',y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[1253]:


'''
#This code block seems to only work when comparing 2 variables. 
import warnings
warnings.filterwarnings('ignore')
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test,Y_test
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()),
                    np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()))
                                      #np.arange(start=X_set[:,2].min()-1,stop=X_set[:,1].max()),
                                      #np.arange(start=X_set[:,3].min()-1,stop=X_set[:,1].max()),
                                      #np.arange(start=X_set[:,4].min()-1,stop=X_set[:,1].max()),
                                      #np.arange(start=X_set[:,5].min()-1,stop=X_set[:,1].max()),
                                      #np.arange(start=X_set[:,6].min()-1,stop=X_set[:,1].max()),
                                      #np.arange(start=X_set[:,7].min()-1,stop=X_set[:,1].max()))
                                                
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1,
                                            alpha=0.75,cmap=ListedColormap('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max()) 
for i, j in enumerate(np.unique(Y_set)):
             plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],
                         c=ListedColormap(('yellow','blue'))(i),label=j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes, Pedigree Function, Age')
plt.ylabel('Outcome')
plt.legend()
plt.show()
'''


# In[1254]:


#Problem 2 


# In[1255]:


#Using Naive Gaussian Bays
from sklearn.naive_bayes import GaussianNB
classifier_G = GaussianNB()
classifier_G.fit(X_train,Y_train)


# In[1256]:


Y2_pred = classifier_G.predict(X_test)


# In[1257]:


Y2_pred


# In[1258]:


conf_matrix = confusion_matrix(Y_test,Y2_pred)
conf_matrix


# In[1259]:


#We are evaluating the model using model evaluation metrics for accuracy, precision, and recall.
print("Accuracy:",metrics.accuracy_score(Y_test,Y2_pred))
print("Precision:",metrics.precision_score(Y_test,Y2_pred))
print("Recall:",metrics.recall_score(Y_test,Y2_pred))


# In[1260]:


#Visualizing model using confusion matrix via matplot.
#Again, we will visualize the confusion matrix using Heatmap.
class_names = [0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)# create heatmap
sns.heatmap(pd.DataFrame(conf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix',y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[1261]:


#Problem 3 Using K-fold cross validation strategy for Logistic Regression


# In[1262]:


#Our Data set will consider Pregnacies through Age as Independent variables (X1-X8) and Outcome as Dependent (Y). 
X=dataset.iloc[:,[0,1,2,3,4,5,6,7]].values
Y=dataset.iloc[:,8].values
#see here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.
X[0:10] #Shows the Array


# In[1263]:


#Doing scaling first
#Doing feature scaling to scale our data between -1 and 1 using standardization.
#Here Scaling is important because there is a significant difference between explanatory variables.
warnings.filterwarnings('ignore')

scalar_X = StandardScaler()
X = scalar_X.fit_transform(X) #New X is scaled
#X_test = scalar_X.transform(X_test) #New X_test is scaled


# In[1264]:


X[0:10] #Shows the Array after scaling


# In[1265]:


Y[0:10] #Shows the Array


# In[1266]:


#Part A n = 5 folds.


# In[1267]:


#Import K-fold cross validation from sklearn.model_selection
#Info gathered at https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

#Doing K-5 folds
from sklearn.model_selection import KFold
X_kfold = X
Y_kfold = Y
kf = KFold(n_splits=5)
kf.get_n_splits(X)

print(kf) 
#n_splits is the number of folds for cross validation
#shuffle is a boolean expression to determine whether the data is shuffled before splitting into folds.
#random_state allows affects the ordering of the indices, which controls the randomness of each fold.

for train_index, test_index in kf.split(X):
    X_train, X_test = X_kfold[train_index], X_kfold[test_index]
    Y_train, Y_test = Y_kfold[train_index], Y_kfold[test_index]
X_train.shape


# In[1268]:


#Running Machine Learning Logistic Regression for K-Folds.
classifier_L.fit(X_train,Y_train)
Y3_pred = classifier_L.predict(X_test)


# In[1269]:


Y3_pred


# In[1270]:


conf_matrix = confusion_matrix(Y_test,Y3_pred)
conf_matrix


# In[1271]:


#We are evaluating the model using model evaluation metrics for accuracy, precision, and recall.
print("Accuracy:",metrics.accuracy_score(Y_test,Y3_pred))
print("Precision:",metrics.precision_score(Y_test,Y3_pred))
print("Recall:",metrics.recall_score(Y_test,Y3_pred))


# In[1272]:


#Visualizing model using confusion matrix via matplot.
#Again, we will visualize the confusion matrix using Heatmap.
class_names = [0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)# create heatmap
sns.heatmap(pd.DataFrame(conf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix',y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[1273]:


#Part B n = 10 folds.


# In[1274]:


#Doing K-10 folds
X_kfold = X
Y_kfold = Y
kf = KFold(n_splits=10)
kf.get_n_splits(X)

print(kf) 
#n_splits is the number of folds for cross validation
#shuffle is a boolean expression to determine whether the data is shuffled before splitting into folds.
#random_state allows affects the ordering of the indices, which controls the randomness of each fold.

for train_index, test_index in kf.split(X):
    X_train, X_test = X_kfold[train_index], X_kfold[test_index]
    Y_train, Y_test = Y_kfold[train_index], Y_kfold[test_index]
X_train.shape


# In[1275]:


#Running Machine Learning Logistic Regression for K-Folds.
classifier_L.fit(X_train,Y_train)
Y3_pred = classifier_L.predict(X_test)


# In[1276]:


Y3_pred


# In[1277]:


conf_matrix = confusion_matrix(Y_test,Y3_pred)
conf_matrix


# In[1278]:


#We are evaluating the model using model evaluation metrics for accuracy, precision, and recall.
print("Accuracy:",metrics.accuracy_score(Y_test,Y3_pred))
print("Precision:",metrics.precision_score(Y_test,Y3_pred))
print("Recall:",metrics.recall_score(Y_test,Y3_pred))


# In[1279]:


#Visualizing model using confusion matrix via matplot.
#Again, we will visualize the confusion matrix using Heatmap.
class_names = [0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)# create heatmap
sns.heatmap(pd.DataFrame(conf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix',y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[1280]:


#Problem 4 Using K-fold cross validation strategy for Naive Bays


# In[1281]:


X[0:10] #Shows the Array


# In[1282]:


Y[0:10] #Shows the Array


# In[1283]:


#Part A n = 5 folds.


# In[1284]:


#Import K-fold cross validation from sklearn.model_selection
#Info gathered at https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

#Doing K-5 folds
from sklearn.model_selection import KFold
X_kfold = X
Y_kfold = Y
kf = KFold(n_splits=5)
kf.get_n_splits(X)

print(kf) 
#n_splits is the number of folds for cross validation
#shuffle is a boolean expression to determine whether the data is shuffled before splitting into folds.
#random_state allows affects the ordering of the indices, which controls the randomness of each fold.

for train_index, test_index in kf.split(X):
    X_train, X_test = X_kfold[train_index], X_kfold[test_index]
    Y_train, Y_test = Y_kfold[train_index], Y_kfold[test_index]
X_train.shape


# In[1285]:


#Running Machine Naive Gaussian Bays for K-Folds.
classifier_G.fit(X_train,Y_train)
Y4_pred = classifier_G.predict(X_test)


# In[1286]:


Y4_pred


# In[1287]:


conf_matrix = confusion_matrix(Y_test,Y4_pred)
conf_matrix


# In[1288]:


#We are evaluating the model using model evaluation metrics for accuracy, precision, and recall.
print("Accuracy:",metrics.accuracy_score(Y_test,Y4_pred))
print("Precision:",metrics.precision_score(Y_test,Y4_pred))
print("Recall:",metrics.recall_score(Y_test,Y4_pred))


# In[1289]:


#Visualizing model using confusion matrix via matplot.
#Again, we will visualize the confusion matrix using Heatmap.
class_names = [0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)# create heatmap
sns.heatmap(pd.DataFrame(conf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix',y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[1290]:


#Part B n = 10 folds


# In[1291]:


#Doing K-10 folds
X_kfold = X
Y_kfold = Y
kf = KFold(n_splits=10)
kf.get_n_splits(X)

print(kf) 
#n_splits is the number of folds for cross validation
#shuffle is a boolean expression to determine whether the data is shuffled before splitting into folds.
#random_state allows affects the ordering of the indices, which controls the randomness of each fold.

for train_index, test_index in kf.split(X):
    X_train, X_test = X_kfold[train_index], X_kfold[test_index]
    Y_train, Y_test = Y_kfold[train_index], Y_kfold[test_index]
X_train.shape


# In[1292]:


#Running Machine Naive Gaussian Bays for K-Folds.
classifier_G.fit(X_train,Y_train)
Y4_pred = classifier_G.predict(X_test)


# In[1293]:


Y4_pred


# In[1294]:


conf_matrix = confusion_matrix(Y_test,Y4_pred)
conf_matrix


# In[1295]:


#We are evaluating the model using model evaluation metrics for accuracy, precision, and recall.
print("Accuracy:",metrics.accuracy_score(Y_test,Y4_pred))
print("Precision:",metrics.precision_score(Y_test,Y4_pred))
print("Recall:",metrics.recall_score(Y_test,Y4_pred))


# In[1296]:


#Visualizing model using confusion matrix via matplot.
#Again, we will visualize the confusion matrix using Heatmap.
class_names = [0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)# create heatmap
sns.heatmap(pd.DataFrame(conf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix',y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

