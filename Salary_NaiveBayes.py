#Importing Libraries 
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
#reading Data
salary_train = pd.read_csv(r"D:\Python\New folder\SalaryData_Train.csv")
salary_test = pd.read_csv(r"D:\Python\New folder\SalaryData_Test.csv")
#Assigning column names to string column variable
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
#transforming string to numbers
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

colnames = salary_train.columns
len(colnames[0:13])
#Train test data
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX  = salary_test[colnames[0:13]]
testY  = salary_test[colnames[13]]
#Applyinh Gaussian and Multinomial NB
sgnb = GaussianNB()
smnb = MultinomialNB()
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) 


spred_mnb = smnb.fit(trainX,trainY)
y=smnb.predict(testX)
mat=confusion_matrix(testY,spred_mnb)
print("Accuracy",(10891+780)/(10891+780+2920+780))  

#Plotting the confusion matrix and NB
import matplotlib.pyplot as plt
import seaborn as sns;
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
            
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.scatter(spred_mnb,testY)
plt.scatter(y,testY)
