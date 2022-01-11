

import pandas as pd
train_set= pd.read_csv("E:\Machine Learning\Assignment\Assignment_1\Training_set_Assignment_1.csv")  

test_set= pd.read_csv("E:\Machine Learning\Assignment\Assignment_1\Test_set_Assignment_1.csv") 
#csv_file["Fire"] = ""
 
print(train_set)

train_set['fire'].unique()
test_set['fire'].unique()
train_set['fire']=train_set['fire'].str.strip()
test_set['fire']=test_set['fire'].str.strip()
test_set['fire'].unique()

#read_file.info()
#train_set.dtypes

train_set.info()
test_set.info()



# Converting categorical variables into dummies/indicator variables

train_set_dummy=pd.get_dummies(data=train_set['fire'])
train_set_dummy
test_set_dummy=pd.get_dummies(data=test_set['fire'])
test_set_dummy


#print(train_set_getdummy.columns)
#print(train_set_getdummy.keys)
#print(test_set_getdummy.columns)
#print(test_set_getdummy.keys)
train_set.iloc[:,0]=train_set_dummy.iloc[:,1]
train_set

test_set.iloc[:,0]=test_set_dummy.iloc[:,1]
test_set
#from sklearn.model_selection import train_test_split

X_train = train_set.drop('fire',axis=1)
y_train = train_set['fire']

X_test = test_set.drop('fire',axis=1)
y_test = test_set['fire']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#--------------------------------------------------------------------

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier



#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)

#Train the model using the training sets
knn.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
predictions_train = knn.predict(X_train)
score_train=accuracy_score(y_train,predictions_train)*100
predictions_test = knn.predict(X_test)
score_test=accuracy_score(y_test,predictions_test)*100
print("The accuracy percentage of KNN for train dataset:",score_train)
print("The accuracy percentage of KNN for test dataset:",score_test)




#----------------------------------------------------------------------
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset


from sklearn.metrics import accuracy_score
predictions_train = gnb.predict(X_train)
score_train=accuracy_score(y_train,predictions_train)*100
predictions_test = gnb.predict(X_test)
score_test=accuracy_score(y_test,predictions_test)*100
print("The accuracy percentage of naive_bayes for train dataset:",score_train)
print("The accuracy percentage of naive_bayes for test dataset:",score_test)

#--------------------------------------------------------------------------

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

no_neighbors = np.arange(1, 20)
train_accuracy = np.empty(len(no_neighbors))
test_accuracy = np.empty(len(no_neighbors))

for i, k in enumerate(no_neighbors):
    # We instantiate the classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
   
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Visualization of k values vs accuracy

plt.title('KNN Vs Accuracy')
#plt.axis(no_neighbors, test_accuracy)
plt.plot(no_neighbors, test_accuracy,'-o', label = 'test data',color ='tab:green')
plt.plot(no_neighbors, train_accuracy,'--o',  label = 'Training data',color ='tab:red')
plt.legend()
plt.xlabel('KNN Value')
plt.ylabel('Accuracy')
plt.show()










