#!/usr/bin/env python
# coding: utf-8

# title : fully-connected neural network & loss drop in different situation 

# 상황 )
#  - one input layer(size 2), one hidden layer(size 8), one output layer(size 1)
#  - learning rate : [0.001,0.005]
#  - stochastic gradient descent : weight update ( batch size = 1 )

#  - (a) plot the **loss&validation loss** versus the number of epochs => convergence of training

# 1. set the environment
import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt('training.txt')
val = np.loadtxt('test.txt') # validation

#  2. define link function as sigmoid

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def dsigmoid(x): # Xin Rong, "word2vec Parameter Learning Explaned", p17
    y = x*(1-x)
    return y

# 3. update equation (train)

def fullNN(n_epochs,lr,input,hidden,output):
    X = train[:,0:2] 
    t = train[:,2:3] #true
    p = np.zeros(1000) #predict

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

# 3. update equation (validation)
def val_fullNN(n_epochs,lr,input,hidden,output):
    valX = val[:,0:2] 
    valt = val[:,2:3] #true
    valp = np.zeros(1000) #predict

    np.random.seed(2020)
    w = np.random.randn(input,hidden) # 2*8
    w_prime = np.random.randn(hidden,output) #8*1
    
    vallossdata = 0 ; vallossarray = []
    
    for epoch in range(n_epochs):
        for i in range(3000):
            pick = np.random.choice(3000, size = 1, replace = False ) #random batch
            
            x = valX[pick] # of input layer
            u = np.dot(x,w) # net input of hidden layer
            
            h = sigmoid(u) # hidden layer
            u_prime = np.dot(h,w_prime) # net output of hidden layer
            
            p = sigmoid(u_prime) # output layer
            
            valloss = 0.5*((p-valt[pick])**2)
            
            """
            chain rule따라 쭉쭉 gradient 계산
            """
            dL_y = p-valt[pick]
            dL_uprime = dL_y*dsigmoid(p)
            dL_h = np.dot(dL_uprime,w_prime.T) #.T : transpose
            dL_u = dL_h*dsigmoid(h)
            dL_w = np.dot(np.transpose(x),dL_u)
            dL_w_prime = np.dot(np.transpose(h),dL_uprime)
                      
            w = w - lr*dL_w
            w_prime = w_prime - lr*dL_w_prime
                                
            vallossdata = vallossdata + np.sum(valloss)
            
        if epoch%100==0:
                print("epoch : "+ str(epoch))
                
        vallossarray.append(vallossdata)
        vallossdata = 0
    return vallossarray

#4. run
n_epochs = 500 ; lr = 0.005 ; input = 2 ; hidden = 8 ; output = 1
lossarray = fullNN(n_epochs,lr,input,hidden,output)
vallossarray = val_fullNN(n_epochs,lr,input,hidden,output)

# 5. output
plt.plot(range(500),lossarray, 'r')
plt.plot(range(500),vallossarray, 'b')
plt.legend(['loss','validation loss'],loc = 1)
plt.title('loss & validation loss per epoch number in SGD')
plt.xlabel('the number of epoch')
plt.ylabel('Loss')
plt.show()


# **validation loss가 loss보다 위에 있음을 확인할 수 있었으며, 유사한 경향을 지니고 떨어지는 상황을 확인할 수 있었습니다.**

#  - [b]  plot the loss versus **the number of hidden layer sizes** 

# 1. set the environment
train = np.loadtxt('training.txt')

# 3. update equation for train
def fullNN_hid(n_epochs,lr,input,hid_st,hid_end,hid_step,output):
    X = train[:,0:2] 
    t = train[:,2:3] #true
    p = np.zeros(1000) #predict

    hiddenlayersize_lossarray=[[],[],[],[],[],[],[],[]]
    for hiddenlayersize in range(hid_st,hid_end,hid_step) :
        
        hiddenlayersize_half = hiddenlayersize/2 - 1
        w = np.random.randn(input,hiddenlayersize) # 2*8
        w_prime = np.random.randn(hiddenlayersize,output) #8*1
        lossdata = 0 ; lossarray = []
        
        for epoch in range(n_epochs):
            
            if epoch%200 == 0:
                print("epoch:", epoch)
                
            for i in range(1000):
                pick = np.random.choice(1000, size = 1, replace = False ) #random batch
            
                x = X[pick] # of input layer
                u = np.dot(x,w) # net input of hidden layer
            
                h = sigmoid(u) # hidden layer
                u_prime = np.dot(h,w_prime) # net output of hidden layer
            
                p = sigmoid(u_prime) # output layer
            
                loss = 0.5*((p-t[pick])**2)
            
                dL_y = p-t[pick]
                dL_uprime = dL_y*dsigmoid(p)
                dL_h = np.dot(dL_uprime,w_prime.T) #.T : transpose
                dL_u = dL_h*dsigmoid(h)
                dL_w = np.dot(np.transpose(x),dL_u)
                dL_w_prime = np.dot(np.transpose(h),dL_uprime)
                      
                w = w - lr*dL_w
                w_prime = w_prime - lr*dL_w_prime
                                
                lossdata = lossdata + np.sum(loss)
            lossarray.append(lossdata)
            lossdata = 0
            
        print("hiddensize : "+ str(hiddenlayersize))
        
        hiddenlayersize_lossarray[int(hiddenlayersize_half)]=lossarray
        lossarray=[]
        # 한 hidden layer size마다 1*500의 lossarray가 만들어졌을 것이다.
        # 각 hidden layer size 별, 즉 8*500의 nd.array를 만들자.  
    return hiddenlayersize_lossarray

#4. run
n_epochs = 500 ; lr = 0.001 ; input = 2 ; hid_st = 2 ; hid_end = 17 ; hid_step = 2 ; output = 1
hiddenlayersize_lossarray = fullNN_hid(n_epochs,lr,input,hid_st,hid_end,hid_step,output)

# 5. output
for i in range(8):
    plt.plot(range(500),hiddenlayersize_lossarray[i])
plt.title('loss per hidden layer sizes')
plt.legend(['hidden layer size of 2','hidden layer size of 4','hidden layer size of 6','hidden layer size of 8','hidden layer size of 10','hidden layer size of 12','hidden layer size of 14','hidden layer size of 16'],loc = 1)
plt.xlabel('the number of epoch')
plt.ylabel('Loss')
plt.show()

# **hidden layer size의 개수에 따라 떨어지는 추이가 조금씩 달랐고, stochastic 방식이라 실행할 때마다 그래프 모양이 달라지기도 했습니다. (혹시나 여러 차례 시행했을 때 동일한 결과를 얻고 싶다면 randomseed를 고정하면 된다는 것도 알게 되었습니다.)**

#  - [c] SGD 대신 Adaptive gradient  : initial rate  = 0.01에 대해서 plot the **loss** versus the number of epochs
# [Adagrad_참고](https://ruder.io/optimizing-gradient-descent/)  
# [Adagrad_이론적 설명](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html)   
# [Adagrad_수식 참고](https://light-tree.tistory.com/140)

# 1. set the environment
train = np.loadtxt('training.txt')

# 3. update equation for train
def fullNN_adagrad(n_epochs,lr,lr2,input,hidden,output,G_t_w,G_t_w_prime):
    X = train[:,0:2] 
    t = train[:,2:3] #true
    p = np.zeros(1000) #predict
    
    w = np.random.randn(input,hidden) # 2*8
    w_prime = np.random.randn(hidden,output) #8*1
    lossdata = 0 ; lossarray = []

    for epoch in range(n_epochs):
        pick = 0
        epsilon =0
        for i in range(1000):
            x = X[pick,np.newaxis] # of input layer
            u = np.dot(x,w) # net input of hidden layer        
            h = sigmoid(u) # hidden layer
            u_prime = np.dot(h,w_prime) # net output of hidden layer        
            p = sigmoid(u_prime) # output layer        
            loss = 0.5*((p-t[pick])**2)
            
            # chain rule따라 쭉쭉 gradient 계산
            dL_y = p-t[pick]
            dL_uprime = dL_y*dsigmoid(p)
            dL_h = np.dot(dL_uprime,w_prime.T) #.T : transpose
            dL_u = dL_h*dsigmoid(h)
            dL_w = np.dot(np.transpose(x),dL_u)
            dL_w_prime = np.dot(np.transpose(h),dL_uprime)
                  
            w = w - lr*dL_w/ np.sqrt(epsilon+G_t_w)
            w_prime = w_prime - lr2*dL_w_prime/ np.sqrt(epsilon+G_t_w_prime)
            """
            (내 생각) initial learningrate가 있으므로, 우선 initial learning rate가 쓰이도록한 후, 
            update한 사항을 다음에 반영해주자
            """
            G_t_w += dL_w*dL_w
            G_t_w_prime += dL_w_prime*dL_w_prime
    
            lossdata = lossdata + np.sum(loss)
            pick = pick + 1
            epsilon = np.exp(-8)
            
        lossarray.append(lossdata)
        lossdata = 0
        
        if epoch%100 == 0:
                print("epoch:", epoch)   
                
    return lossarray

# 4. run
n_epochs = 500 ; lr = 0.01 ; lr2 = 0.01 ; input = 2 ; hidden = 8 ; output = 1 ; G_t_w = 1;G_t_w_prime = 1
lossarray = fullNN_adagrad(n_epochs,lr,lr2,input,hidden,output,G_t_w,G_t_w_prime)

# 5. output
plt.plot(range(n_epochs),lossarray)
plt.title('loss per the number of epoch [Adagrad]')
plt.xlabel('the number of epoch')
plt.ylabel('Loss [Adagrad]')
plt.show()

# **굉장히 빠른 속도로 loss가 떨어지는 것을 확인할 수 있었다. 당연한 소리이겠지만, SGD와는 달리 stochastic한 모습이 잘 보이지 않았다.**