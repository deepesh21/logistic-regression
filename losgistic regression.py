#!/usr/bin/env python
# coding: utf-8

# In[160]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[161]:


x=np.array([0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,3.75,4.00,4.25,4.50,4.75,5.00,5.50])
y=np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,2,1,1,1,1,1])


# In[162]:


plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('regression')
plt.show()


# In[163]:


x=np.c_[np.ones(x.shape[0]),x]


# In[164]:


alpha=0.00001
m=y.size
np.random.seed(10)
theta=np.random.rand(2)
theta.shape
z=np.dot(x,theta.T)
prediction=1/(1+np.exp(-z))
prediction
step1=np.multiply(y,np.log(prediction))
step2=np.multiply(np.subtract(1,y),np.log(np.subtract(1,prediction)))
step3=-step1 - step2
cost=np.mean(step3)
theta-alpha*np.dot(np.subtract(prediction,y).T,x)

x.shape


# In[222]:


def gradient_des(x,y,alpha,theta,m):
    cost_list = []   
    theta_list = []  
    prediction_list = []
    run = True
    cost_list.append(1e10)    
    i=0
    while run:
        z=np.dot(x,theta.T)
        prediction=1/(1+np.exp(-z))
        print(prediction)
        prediction_list.append(prediction)
        step1=np.multiply(y,np.log(prediction))
        step2=np.multiply(np.subtract(1,y),np.log(np.subtract(1,prediction)))
        step3=-step1 - step2
        cost=np.mean(step3)
        cost_list.append(cost)
        theta=theta-alpha*np.dot(np.subtract(prediction,y).T,x)
        theta_list.append(theta)
        if cost_list[i]-cost_list[i+1] < 1e-9:
            run=False
        i+=1
    cost_list.pop(0)
    return prediction_list,theta_list,cost_list


# In[223]:


prediction_list, theta_list, cost_list = gradient_des(x, y, alpha, theta, m)


# In[109]:


plt.plot(cost_list)
plt.xlabel('x')
plt.ylabel('y')
plt.title("cost function")
plt.show()


# In[169]:


ax1=plt.subplot(121)
plt.xlabel('x')
plt.ylabel('y')
plt.title("data points")
ax1.scatter(x[:,1],y,color='C1')
plt.subplot(122)
plt.plot(prediction_list[-1])
plt.xlabel('x')
plt.ylabel('y')
plt.title("regression line")
plt.show()

