---
layout: post
title:  "사이킷런 없이 로지스틱 회귀 구현하기()"
---


```python
# 준비
import sys
assert sys.version_info >= (3, 5)
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
```


```python
#데이터세트 준비
from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"][:, 3:]                   
y = (iris["target"] == 2).astype(np.int)  

X_with_bias = np.c_[np.ones([len(X),1]),X]

np.random.seed(2042)
```


```python
#데이터분할-사이킷런 train_test_spilit 구현
test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size

rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]

X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]

def to_one_hot(y):
    n_classes = 2                # 클래스 수를 2로 고정시키면 로지스틱 함수와 같다.
    m = len(y)                             
    Y_one_hot = np.zeros((m, n_classes))    
    Y_one_hot[np.arange(m), y] = 1          
    return Y_one_hot

Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)

ytr=Y_train_one_hot[:,1]
yva=Y_valid_one_hot
yte=Y_test_one_hot
```


```python
#로지스틱 회귀는 시그모이드 함수 사용
#SIGMOID FUNCTION
def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head
```


```python
#배치경사하강법 활용
n_inputs = X_train.shape[1]           
n_outputs = len(np.unique(y_train))   

Theta = np.random.randn(n_inputs, n_outputs)

eta = 0.01
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7

for iteration in range(n_iterations):     
    logits = X_train.dot(Theta)
    Y_proba = sigmoid(logits)
    
    if iteration % 500 == 0:              
        loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
        print(iteration, loss)
    
    error = Y_proba - Y_train_one_hot     
    gradients = 1/m * X_train.T.dot(error)
    
    Theta = Theta - eta * gradients   
```

    0 0.6216585388989472
    500 0.5358155407854331
    1000 0.46868915858417043
    1500 0.4228110723139094
    2000 0.389997119276909
    2500 0.36532453055572156
    3000 0.34600634309210915
    3500 0.33038385603797926
    4000 0.3174181616594949
    4500 0.3064286658965357
    5000 0.2969519170535297
    


```python
logits = X_valid.dot(Theta)              
Y_proba = sigmoid(logits)
y_predict = np.argmax(Y_proba, axis=1)          

accuracy_score = np.mean(y_predict == y_valid)  
accuracy_score
```




    0.9666666666666667




```python
#규제 더한 배치경사하강법

eta = 0.1
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1       

Theta = np.random.randn(n_inputs, n_outputs)  #Theta 리셋

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = sigmoid(logits)
    
    if iteration % 500 == 0:
         xentropy_loss =  -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
         l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
         loss = (xentropy_loss + alpha * l2_loss)    
         print(iteration, loss)
    
    error = Y_proba - Y_train_one_hot
    l2_loss_gradients = np.r_[np.zeros([1, n_outputs]), alpha * Theta[1:]]   
    gradients = 1/m * X_train.T.dot(error) + l2_loss_gradients
    
    Theta = Theta - eta * gradients
```

    0 0.7162610260856582
    500 0.5424422902989773
    1000 0.5489288755349917
    1500 0.5496400244557951
    2000 0.5497072521674562
    2500 0.5497135348935831
    3000 0.5497141214229124
    3500 0.5497141761734806
    4000 0.5497141812842178
    4500 0.5497141817612832
    5000 0.5497141818058153
    


```python
logits = X_valid.dot(Theta)
Y_proba = sigmoid(logits)
y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_valid)
accuracy_score
```




    0.9




```python
#조기 종료 추가
eta = 0.1 
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1            
best_loss = np.infty   

Theta = np.random.randn(n_inputs, n_outputs)  

for iteration in range(n_iterations):
    
    logits = X_train.dot(Theta)
    Y_proba = sigmoid(logits)
    error = Y_proba - Y_train_one_hot
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), alpha * Theta[1:]]
    Theta = Theta - eta * gradients

    logits = X_valid.dot(Theta)
    Y_proba = sigmoid(logits)
    xentropy_loss = -np.mean(np.sum(Y_valid_one_hot * np.log(Y_proba + epsilon), axis=1))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
    loss = (xentropy_loss + alpha * l2_loss)
    

    if iteration % 500 == 0:
        print(iteration, loss)
        
    
    if loss < best_loss:
        best_loss = loss
    else:                                      
        print(iteration - 1, best_loss)        
        print(iteration, loss, "조기 종료!")
        break

```

    0 0.6586122784654648
    0 0.6586122784654648
    1 0.6614977663257443 조기 종료!
    


```python
logits = X_valid.dot(Theta)
Y_proba = sigmoid(logits)
y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_valid)
accuracy_score
```




    0.3333333333333333


