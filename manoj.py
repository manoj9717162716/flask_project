#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


data=pd.read_csv("50_Startups.csv")


# In[4]:


data=data.drop('State',axis=1)


# In[5]:


x=data.iloc[:,0:-1]


# In[6]:


y=data.iloc[:,-1]


# In[7]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)


# In[8]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)


# In[9]:


ypred=model.predict(xtest)


# In[10]:


from sklearn.metrics import r2_score
r2_score(ytest,ypred)


# In[25]:


from flask import Flask, render_template, request
manoj=Flask(__name__)
@manoj.route('/')
def xyz():
    return render_template("web.html")
@manoj.route('/detail',methods=['GET','POST'])
def abc():
    if(request.method=='POST'):
        a=int(request.form['v1'])
        b=int(request.form['v2'])
        c=int(request.form['v3'])
        pred=model.predict([[a,b,c]])
        return render_template('web.html',result=pred)
if('__main__')==__name__:
    manoj.run()

