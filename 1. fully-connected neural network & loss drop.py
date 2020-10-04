#!/usr/bin/env python
# coding: utf-8

# title : fully-connected neural network & loss drop

# 상황 )
#  - one input layer(size 2), one hidden layer(size 8), one output layer(size 1)
#  - learning rate : 0.005
#  - stochastic gradient descent : weight update ( batch size = 1 )
#  - plot the loss versus the number of epochs => convergence of training

# 1. set the environment

import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt('training.txt')
X = train[:,0:2] 
t = train[:,2:3] #true
p = np.zeros(1000) #predict

#  2. define link function as sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):   # 'word2vec' p17
    return x*(1-x)

# 3. update equation
def fullNN(n_epochs,lr,input,hidden,output):
    np.random.seed(2020)
    w = np.random.randn(input,hidden) # 2*8
    w_prime = np.random.randn(hidden,output) #8*1
    
    lossdata = 0 ; lossarray = []
    
    for epoch in range(n_epochs):
        for i in range(1000):
            pick = np.random.choice(1000, size = 1, replace = False ) #random batch
            
            x = X[pick] # of input layer
            u = np.dot(x,w) # net input of hidden layer
            
            h = sigmoid(u) # hidden layer
            u_prime = np.dot(h,w_prime) # net output of hidden layer
            
            p = sigmoid(u_prime) # output layer
            
            loss = 0.5*((p-t[pick])**2)
            
            """
            chain rule따라 쭉쭉 gradient 계산
            """
            dL_y = p-t[pick]
            dL_uprime = dL_y*dsigmoid(p)
            dL_h = np.dot(dL_uprime,w_prime.T) #.T : transpose
            dL_u = dL_h*dsigmoid(h)
            dL_w = np.dot(np.transpose(x),dL_u)
            dL_w_prime = np.dot(np.transpose(h),dL_uprime)
                      
            w = w - lr*dL_w
            w_prime = w_prime - lr*dL_w_prime
            lossdata = lossdata + np.sum(loss)
            
        if epoch%100==0:
            print("epoch : "+ str(epoch))
            
        lossarray.append(lossdata)
        lossdata = 0
    return lossarray

# 4. run
n_epochs = 500 ; lr = 0.005 ; input = 2 ; hidden = 8 ; output = 1
lossarray = fullNN(n_epochs,lr,input,hidden,output)

# 5. output
plt.plot(range(500),lossarray)
plt.title('loss per epoch number in SGD')
plt.xlabel('the number of epoch')
plt.ylabel('Loss')
plt.show()