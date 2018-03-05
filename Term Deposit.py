
"""
/*****************************************************************************************************************
*****************************************************************************************************************/
    > AUTHOR: Diogo Santos
        
        > Summary: It comes from UCI Machine Learning for marketing campaings of a Portuguese Bank.
                   The main goal is to predict a term loan subscrition from the client.
          
        > Data: https://archive.ics.uci.edu/ml/datasets/bank+marketing
                Here is all information about the data, includind Download and Variables Description
    
/*****************************************************************************************************************
*****************************************************************************************************************/
"""

#Importing Relevant Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

data = pd.read_csv(r'C:\Users\dsilsant\Downloads\DS_dataset\term loan deposits.csv')

data.head()
data.info()

#1 - EXPLORATORY DATA ANALYSIS #
    #Target Variable - y
    
f,ax=plt.subplots(1,2,figsize=(15,8))
data['y'].value_counts().plot.pie(explode =[0,0.1],autopct='%1.1f%%',ax=ax[0])
ax[0].set_title('term deposit')
sns.countplot('y',data=data,ax=ax[1])
ax[0].set_title('term deposit')

    #Analyzing features
    
        #AGE
        
f,ax=plt.subplots(2,1, figsize=(15,10))
sns.distplot(data.loc[data['y']==0,'age'],ax=ax[0])
ax[0].set_title('Age Distribution for no Term Loan')
sns.distplot(data.loc[data['y']==1,'age'],ax=ax[1])
ax[1].set_title('Age Distribution for Term Loan')

'''COMMENT = 1) More concentrated population between 20 and 40 subscribing the product
             2) People with more than 60 year old, are very likely to subscribe'''

        #JOB

data.groupby(['job'])['y'].mean()

f,ax = plt.subplots(2,1, figsize=(20,15))
pd.crosstab(data.job,data.y).plot(kind='bar',ax=ax[0])
sns.factorplot('job','y',data=data,ax=ax[1])
plt.close()

'''COMMENT = 1) Most part of people in dataset is admin,blue-color, and technician
             2) Students and retired people tend to subscribe more term-deposits'''

        #MARITAL
#Dealing with unknown values

marital_unkwown=data.loc[data['marital']=='unknown',['education','job','age']] # To create a prediction of unknown marital status
marital_unkwown.describe()
marital_unkwown.describe(include = ['object']) #getting the means and top obtained
marital_unkwown=data.loc[(data['education']=='university.degree')&(data['job']=='blue-collar')&(data['age']>=32)&(data['age']<=42),] #apllying the results from the mean and top
marital_unkwown.describe(include = ['object']) # the unkwown people tend to be single based on education, job and age
data.loc[data['marital']=='unknown','marital']='single'

data.groupby(['marital'])['y'].mean()

f,ax=plt.subplots(1,2, figsize=(14,5))
pd.crosstab(data.marital,data.y).plot(kind='bar', ax=ax[0])
sns.factorplot('marital','y',data=data, ax=ax[1])
plt.close()

'''COMMENT = 1) Single tend to be the group subscribing more term loans'''
        
        #EDUCATION
data['education'].unique()

#Lest convert all the basic from 4y,6y and 9y into only basic education
data.groupby(['education'])['y'].mean()
data.loc[data['education']=='basic.4y','education']='basic'
data.loc[data['education']=='basic.6y','education']='basic'
data.loc[data['education']=='basic.9y','education']='basic'

#Lest now uncover the unknown values
educationdata=data.loc[data['education']=='unknown',]
educationdata.describe()
educationdata.describe(include=['object'])
educationdata=data.loc[(data['marital']=='married')&(data['job']=='blue-collar')&(data['age']>=38)&(data['age']<=46),]
educationdata.describe(include=['object'])
data.loc[data['education']=='unknown','education']='basic'

f,ax=plt.subplots(2,1,figsize=(10,8))
pd.crosstab(data.education,data.y).plot(kind='bar',ax=ax[0])
sns.factorplot('education','y',data=data,size=12,ax=ax[1])
plt.close()

'''COMMENT = 1) Education doesnt' seem to be a strong variable predicting the subscription of term loans'''

        #DEFAULT
        
data.groupby(['default'])['y'].mean()
data['default'].value_counts()
pd.crosstab(data.default,data.loan,margins=True)
data.loc[data['default']=='unknown','default']='no'
pd.crosstab(data.default,data.y).plot(kind='bar')

'''COMMENT = 1) People with no default are the ones with term loan deposits
             2) Makes sense since they have no late payments'''

        #HOUSING
data.groupby(['housing'])['y'].mean()
housingdata=data.loc[data['housing']=='unknown',]
housingdata.describe(include=['object'])      
pd.crosstab(data.housing,data.y).plot(kind='bar')
sns.factorplot('housing','y',data=data)

'''COMMENT = 1) Not relevant for thr analysis'''

        #LOAN
pd.crosstab(data.loan,data.y,margins=True)
pd.crosstab([data.loan,data.default],[data.y],margins = True)
pd.crosstab([data.loan,data.housing],[data.y],margins = True)
pd.crosstab(data.loan,data.y).plot(kind='bar')  

'''COMMENT = 1) Individuals with no loans are the ones doing term loans'''

        #CONTACT
pd.crosstab(data.contact,data.y,margins=True)
pd.crosstab([data.contact,data.default],[data.y],margins=True)
sns.factorplot('contact','y',hue='default',data=data)

'''COMMENT = 1) Individuals being contacted using cellular are the ones doing term loans
             2) being contacted with cellular for people with no default seems to predict very well the term loan'''

        #MONTH
pd.crosstab(data.month,data.y,margins=True)
sns.factorplot('month','y',data=data,size=8)

pd.crosstab(data.day_of_week,data.y,margins=True)
pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
pd.crosstab([data.day_of_week,data.default],[data.y],margins=True)
sns.factorplot('default','y',col='day_of_week',data=data,size=8)

'''COMMENT = 1) Better days for contact are thu,tue and wed'''

        #CAMPAIGN
data.campaign.plot(kind='hist',bins=50)
data['campaign'].describe()

campaign = data.loc[data['campaign']< 12,['campaign','y']]

pd.crosstab(campaign.campaign,campaign.y,margins=True)
pd.crosstab(campaign.campaign,campaign.y).plot(kind='bar')
sns.factorplot('campaign','y',data=campaign,size=10)
            
'''COMMENT = 1) One should do only until 5 campaigns for the client to subscribe a term loan deposit''' 

        #PDAYS
data['pdays'].describe()
pdays=data.loc[data['pdays']!=999,['campaign','y','pdays']]
pdays['pdays'].describe()
pdays.pdays.plot(kind='hist')

pd.crosstab(pdays.pdays,pdays.y).plot(kind='bar',figsize=(15,10))

'''COMMENT = 1) Most part of clients subscribe the deposit 3rd and 6th day after the last contact'''

        #PREVIOUS
data['previous'].unique()
pd.crosstab(data.previous,data.y,margins=True)

f,ax=plt.subplots(2,1,figsize=(15,10))
pd.crosstab(data.previous,data.y).plot(kind='bar',ax=ax[0])
sns.factorplot('previous','y',data=data,ax=ax[1])
plt.close()

'''COMMENT = 1) Clients who subscribed weren't even contacted before'''

        #POUTCOME
pd.crosstab(data.poutcome,data.y,margins=True)
f,ax=plt.subplots(2,1,figsize=(15,10))
pd.crosstab(data.poutcome,data.y).plot(kind='bar',ax=ax[0])
sns.factorplot('poutcome','y',data=data,ax=ax[1])
plt.close()


'''COMMENT = 1) The sucess of previous campaings favours the new one'''

#1 - DATA ENGEENIRING ###################################

#Age
data['age'].describe()
print(97/17)

data.loc[data['age']>20,'age'] = 0
data.loc[(data['age']>20)&(data['age']<=32),'age']=1
data.loc[(data['age']>32)&(data['age']<=48),'age']=2
data.loc[(data['age']>48)&(data['age']<=64),'age']=3
data.loc[data['age']>64,'age']=4

#JOB
data['job'].unique()

data.loc[data['job']=='blue-collar','job']= 0
data.loc[data['job']=='technician','job']= 1
data.loc[data['job']=='management','job']= 2
data.loc[data['job']=='services','job']= 3
data.loc[data['job']=='retired','job']= 4
data.loc[data['job']=='technician','job']= 5
data.loc[data['job']=='admin.','job']= 6
data.loc[data['job']=='housemaid','job']= 6
data.loc[data['job']=='unemployed','job']= 6
data.loc[data['job']=='entrepreneur','job']= 6
data.loc[data['job']=='self-employed','job']= 6
data.loc[data['job']=='unknown','job']= 6
data.loc[data['job']=='student','job']= 6

#MARITAL
data['marital'].unique()

data.loc[data['marital']=='divorced','marital']= 0
data.loc[data['marital']=='married','marital']= 1
data.loc[data['marital']=='single','marital']= 2

#EDUCATION
data['education'].unique()

data.loc[data['education']=='basic','education']=0
data.loc[data['education']=='high.school','education']=1
data.loc[data['education']=='university.degree','education']=2
data.loc[data['education']=='illiterate','education']=3
data.loc[data['education']=='professional.course','education']=3

#DEFAULT
pd.crosstab(data.default,data.y,margins=True)
pd.crosstab(data.default,data.loan,margins=True) #unknown seems to be people with no loans meaning they're not default, as they're also having a term loan

data.loc[data['default']=='no','default']=0
data.loc[(data['default']=='yes') | (data['default']=='unknown'),'default']=1
data['default'].value_counts()

#PDAYS
data['pdays'].describe()
pd.crosstab(data.pdays,data.y,margins=True)

data.loc[data['pdays']<5,'pdays']=0
data.loc[(data['pdays']>=5)&(data['pdays']<10),'pdays']=1
data.loc[(data['pdays']>=10)&(data['pdays']<15),'pdays']=2
data.loc[(data['pdays']>=15)&(data['pdays']<30),'pdays']=3
data.loc[data['pdays']==999,'pdays']=4

pd.crosstab(data.pdays,data.y,margins=True)

#CONVERTING STRINGS TO NUMERIC
data['housing'].replace(['yes','no','unknown'],[0,1,2],inplace=True)
data['loan'].replace(['yes','no','unknown'],[0,1,0],inplace=True)
data['contact'].replace(['cellular','telephone'],[0,1],inplace=True)
data['month'].replace(['may','jul','aug','jun','nov','apr','oct','sep','mar','dec'],[0,1,2,3,4,5,6,7,8,9],inplace=True)
data['day_of_week'].replace(['thu','mon','wed','tue','fri'],[0,1,2,3,4],inplace=True)
data['poutcome'].replace(['nonexistent', 'success', 'failure'],[0,1,2],inplace = True)


# MULTICOLLINEARITY AND CORRELATIONS
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(x):
    thresh = 10.0
    output = pd.DataFrame()
    vif = [variance_inflation_factor(np.array(X.values,dtype='float'), j) for j in range(X.shape[1])]
    for i in range(1,X.shape[1]):
        print("Iteration no.: ", i)
        print(vif)
        a = np.argmax(vif)
        print("Max VIF is for variable no.:",X.columns[a])
        print("With the value of:", vif[a])
        print( )
        if vif[a] <= thresh :
            break
        if i == 1 :          
            output = X.drop(X.columns[a], axis = 1)
            vif = [variance_inflation_factor(np.array(output.values,dtype='float'), j) for j in range(output.shape[1])]
        elif i > 1 :
            output = output.drop(output.columns[a],axis = 1)
            vif = [variance_inflation_factor(np.array(output.values,dtype='float'), j) for j in range(output.shape[1])]
    return(output.columns)

calculate_vif(X)
  
#3 - MODELLING ###################################

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV #Hyper_Parameters Tuning

X=data.loc[:,['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'previous',
       'poutcome', 'emp_var_rate']]
y=data['y']

train_x,test_x,train_y,test_y = train_test_split(X,y,random_state=0, test_size=0.3)

from sklearn import svm
model=svm.SVC()
model.fit(train_x,train_y)
prediction=model.predict(test_x)
print('Accuracy.: ',metrics.accuracy_score(prediction,test_y))

model=svm.SVC(kernel='linear')
model.fit(train_x,train_y)
prediction1=model.predict(test_x)
print('Accuracy.: ',metrics.accuracy_score(prediction1,test_y))

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
rfe = RFE(model)
rfe = rfe.fit(X,y)
print(rfe.support_)
print(rfe.ranking_)

model=LogisticRegression()
model.fit(train_x,train_y)
print('Intercept is : ',model.intercept_,'\n',
      'The coefficients are :','\n',
      pd.concat([pd.DataFrame(train_x.columns),pd.DataFrame(np.transpose(model.coef_)),pd.DataFrame(np.transpose(rfe.ranking_))],axis=1))
prediction2=model.predict(test_x)
print('Accuracy.: ',metrics.accuracy_score(prediction2,test_y))

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(train_x,train_y)
prediction4=model.predict(test_x)
print('Accuracy.: {}'.format(metrics.accuracy_score(prediction4,test_y)))

b_index=list(range(10,30))
b=pd.Series()
for i in b_index:
    model=DecisionTreeClassifier(max_leaf_nodes=i)
    model.fit(train_x,train_y)
    prediction=model.predict(test_x)
    b=b.append(pd.Series((metrics.accuracy_score(prediction,test_y))))
plt.subplots(figsize=(15,10))
plt.plot(b_index,b)
plt.xlabel('Nodes')
plt.ylabel('Accuracy')
plt.title('Tree vs Accuracy')

treemodel=pd.DataFrame({'nodes':b_index,'values':b.values}) # to get the best parameter
treemodel.loc[treemodel['values']==b.values.max(),]

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(train_x,train_y)
prediction5=model.predict(test_x)
print('Accuracy.: {}'.format(metrics.accuracy_score(prediction5,test_y)))

a_index=list(range(5,20))
a=pd.Series()
for i in a_index:
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(train_x,train_y)
    prediction=model.predict(test_x)
    a=a.append(pd.Series((metrics.accuracy_score(prediction,test_y))))
plt.subplots(figsize=(15,10))
plt.plot(a_index,a)
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')

Knnmodel=pd.DataFrame({'neighbors':a_index,'values':a.values}) # to get the best parameter
Knnmodel.loc[Knnmodel['values']==a.values.max(),]

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(train_x,train_y)
prediction6=model.predict(test_x)
print('Accuracy.: {}'.format(metrics.accuracy_score(prediction6,test_y)))

from sklearn.neural_network import MLPClassifier
model=MLPClassifier()
model.fit(train_x,train_y)
prediction7=model.predict(test_x)
print('Accuracy.: {}'.format(metrics.accuracy_score(prediction7,test_y)))

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(train_x,train_y)
prediction3=model.predict(test_x)
print('Accuracy.: ',metrics.accuracy_score(prediction3,test_y))

index=list(range(20,40))
rf=pd.Series()
for i in index:
    model=RandomForestClassifier(max_leaf_nodes=i)
    model.fit(train_x,train_y)
    prediction=model.predict(test_x)
    rf=rf.append(pd.Series(metrics.accuracy_score(prediction,test_y)))
rfmodel=pd.DataFrame({'nodes':index,'values':rf.values})
plt.subplots(figsize=(15,10))
plt.plot(index,rf)
plt.xlabel('nodes')

rfmodel.loc[rfmodel['values']==rf.values.max(), ] # to get the best parameter

from sklearn.ensemble import AdaBoostClassifier
model=AdaBoostClassifier()
model.fit(train_x,train_y)
prediction8=model.predict(test_x)
print('Accuracy.: ',metrics.accuracy_score(test_y,prediction8))

ada_index=[50,500,1000,5000]
ada=pd.Series()
for i in ada_index:
    model=AdaBoostClassifier(n_estimators=i)
    model.fit(train_x,train_y)
    prediction=model.predict(test_x)
    ada=ada.append(pd.Series(metrics.accuracy_score(test_y,prediction)))   
ada_model=pd.DataFrame({'Estimators':ada_index,'Values':ada.values})
plt.plot(ada_index,ada)
plt.xlabel('estimators')
plt.ylabel('accuracy')

from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()
model.fit(train_x,train_y)
prediction9=model.predict(test_x)
print('Accuracy.: ',metrics.accuracy_score(test_y,prediction9))

n_estimators=list(range(100,1600,100))
learning_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learning_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper)
gd.fit(train_x,train_y)
gd.best_score_
gd.best_estimator_


#4 - VALIDATION ###################################

#CROSS VALIDATION

from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation

kfold=KFold()
accuracy_mean=[]
accuracy_std=[]
accuracy=[]
classifiers=['rbf svm','linear svm','logistic regression','decision trees','random forest',
             'kkneighbors','naive bayes','neural networks','adaboostclassifier','GradientBoostingClassifier']
models=[svm.SVC(),svm.SVC(kernel='linear'),LogisticRegression(),DecisionTreeClassifier(max_leaf_nodes=20),
        RandomForestClassifier(max_leaf_nodes=39),KNeighborsClassifier(n_neighbors=17),GaussianNB(),MLPClassifier()]
for i in models:
    model=i
    cv_result=cross_val_score(model,X,y,cv=kfold,scoring='accuracy')
    accuracy_mean.append(cv_result.mean())
    accuracy_std.append(cv_result.std())

model_frame=pd.DataFrame({'classifiers':classifiers,'Model_Mean':accuracy_mean,'Model_Std':accuracy_std})
pd.DataFrame(accuracy_mean,index=classifiers).plot(kind='bar',color='orange')

# CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict #prediction

y_pred=cross_val_predict(DecisionTreeClassifier(max_leaf_nodes=20),X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),annot=True,fmt='2.0f')

#FEATURE IMPORTANCE FOR THE BEST MODEL
model=DecisionTreeClassifier(max_leaf_nodes=15)
model.fit(X,y)  
pd.Series(model.feature_importances_,X.columns).sort_values().plot(kind='barh',width=0.8,title='feature importance in Decision Trees')


