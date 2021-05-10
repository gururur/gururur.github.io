---
layout: post
title:  "RE.사이킷런 없이 로지스틱 회귀 구현하기-성공"
---

원 핫 벡터는 소프트맥스에서와는 달리 로지스틱 회귀에서 오류를 유발함


```python
#과제 1 : 사이킷런의 도움 없이 로지스틱 회귀 구현하기
#준비

import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 깔끔한 그래프 출력
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 그림을 저장 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training_linear_models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
# 어레이 데이터를 csv 파일로 저장
def save_data(fileName, arrayName, header=''):
    np.savetxt(fileName, arrayName, delimiter=',', header=header, comments='')
```


```python
#데이터 준비

from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]        #특성 2,3번에 해당하는 꽃잎 길이와 너비            
y = (iris["target"] == 2).astype(np.int)  #virginica 판단 모델용 데이터셋

X_with_bias = np.c_[np.ones([len(X),1]),X]  #0번특성 x0 판단 막기위한 편향 추가

np.random.seed(2042)  #랜덤 시드 지정하여 같은 결과 유도

#데이터를 훈련세트 60%, 검증세트 20%, 테스트 세트 20%로 분할
test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size

rnd_indices = np.random.permutation(total_size)  #인덱스를 무작위로 섞는 함수

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]

X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]
```


```python
#로지스틱 회귀는 시그노이드 함수를 사용함
def sigmoid(z):
    y_head = 1.0 / (1 + np.exp(-z))
    return y_head
```


```python
#가중치 조정 위한 편향의 특성과 갯수만큼 세타 랜덤초기화
n_inputs = X_train.shape[1]
Theta = np.random.randn(n_inputs)
```


```python
#  배치경사하강법
eta = 0.1
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7

for iteration in range(n_iterations):   
    logits = X_train.dot(Theta)
    Y_proba = sigmoid(logits)
   
    if iteration % 500 == 0:
      loss = -np.mean(np.sum((y_train*np.log(Y_proba + epsilon) + (1-y_train)*np.log(1 - Y_proba + epsilon))))  #소프트맥스 비용함수와 비슷하지 않음 주의할 것
      print(iteration, loss)
    
    error = Y_proba - y_train     # 그레이디언트=기울기 계산.
    gradients = 1/m * X_train.T.dot(error)
    
    Theta = Theta - eta * gradients
```

    0 79.35473984499612
    500 27.149524631560638
    1000 21.89438928577945
    1500 19.33777344771706
    2000 17.69144423932671
    2500 16.49516908325313
    3000 15.566000472955372
    3500 14.813273989795578
    4000 14.185530546071131
    4500 13.65075154805576
    5000 13.187653637231024
    


```python
#검증 : 로지스틱 회귀에서 0.5 이상이냐 아니냐를 따져 1과 0으로 구분하는 작업을 여기서 구현, 이후 검증세트와 비교해서 정확도를 확인함
logits = X_valid.dot(Theta)              
Y_proba = sigmoid(logits)
y_predict = np.array([])
for i in range(len(Y_proba)): #0.5 이상이냐 아니냐를 따짐
  if Y_proba[i] >= 0.5:
    y_predict = np.append(y_predict, 1)
  else:
    y_predict = np.append(y_predict, 0)

accuracy_score = np.mean(y_predict == y_valid) 
accuracy_score
```




    0.9666666666666667




```python
#규제를 더한 배치경사하강법

eta = 0.1
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1        # 규제 하이퍼파라미터

Theta = np.random.randn(n_inputs) #앞에서 쓴 세타를 그대로 쓸수 없으므로 새로이 초기화함

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = sigmoid(logits)
    
    if iteration % 500 == 0:
        xentropy_loss = -np.mean(np.sum((y_train*np.log(Y_proba + epsilon) + (1-y_train)*np.log(1 - Y_proba + epsilon))))
        l2_loss = 1/2 * np.sum(np.square(Theta[1:]))  # 편향은 규제에서 제외
        loss = xentropy_loss + alpha * l2_loss        # l2 규제가 추가된 손실
        print(iteration, loss)
    
    error = Y_proba - y_train
    l2_loss_gradients = np.r_[np.zeros([1]), alpha * Theta[1:]]   # l2 규제 그레이디언트
    gradients = 1/m * X_train.T.dot(error) + l2_loss_gradients
    
    Theta = Theta - eta * gradients
```

    0 111.90695597427701
    500 31.541062387070156
    1000 27.24748939312895
    1500 25.53371922045158
    2000 24.633911368261604
    2500 24.105826194343813
    3000 23.777679200466416
    3500 23.56698006866457
    4000 23.428957737110814
    4500 23.33738818615023
    5000 23.27613392426706
    


```python
#다시 검증
logits = X_valid.dot(Theta)              
Y_proba = sigmoid(logits)
y_predict = np.array([])
for i in range(len(Y_proba)):
  if Y_proba[i] >= 0.5:
    y_predict = np.append(y_predict, 1)
  else:
    y_predict = np.append(y_predict, 0)

accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산
accuracy_score
```




    0.9333333333333333




```python
#조기 종료 + 규제 + 배치경사하강법
#조기종료는 검증세트에 대한 손실값이 이전 단계보다 커지면 종료되는 기능
eta = 0.1 
n_iterations = 50000
m = len(X_train)
epsilon = 1e-7
alpha = 0.1            
best_loss = np.infty   # 최소 손실값 기억 변수

Theta = np.random.randn(n_inputs)

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta)
    Y_proba = sigmoid(logits)
    error = Y_proba - y_train
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1]), alpha * Theta[1:]]
    Theta = Theta - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta)
    Y_proba = sigmoid(logits)
    xentropy_loss = -np.mean(np.sum((y_valid*np.log(Y_proba + epsilon) + (1-y_valid)*np.log(1 - Y_proba + epsilon))))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되기 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```

    0 41.29329025025703
    500 11.142702476300443
    1000 9.607704746493344
    1500 9.034856779683722
    2000 8.744020504182084
    2500 8.576677705056653
    3000 8.473981809617236
    3500 8.408575411053302
    4000 8.365959554422538
    4500 8.337787983559863
    5000 8.318988514505117
    5500 8.306364843349028
    6000 8.297852861288574
    6500 8.292097302584537
    7000 8.28819823066339
    7500 8.285553463957891
    8000 8.283757954624514
    8500 8.282538286255761
    9000 8.281709451126162
    9500 8.281146057959104
    10000 8.280763026422438
    10500 8.280502584105157
    11000 8.280325481347905
    11500 8.280205043195352
    12000 8.280123136417528
    12500 8.280067432318521
    13000 8.280029547751669
    13500 8.280003781998426
    14000 8.27998625825057
    14500 8.27997433996958
    15000 8.279966234056804
    15500 8.27996072101385
    16000 8.279956971442665
    16500 8.27995442125371
    17000 8.279952686796952
    17500 8.27995150714243
    18000 8.27995070482474
    18500 8.27995015914478
    19000 8.27994978801165
    19500 8.279949535592976
    20000 8.279949363915506
    20500 8.279949247152537
    21000 8.279949167738554
    21500 8.279949113726726
    22000 8.27994907699167
    22500 8.279949052007069
    23000 8.279949035014294
    23500 8.279949023456993
    24000 8.279949015596527
    24500 8.279949010250393
    25000 8.279949006614325
    25500 8.279949004141326
    26000 8.279949002459364
    26500 8.279949001315408
    27000 8.279949000537373
    27500 8.27994900000821
    28000 8.279948999648306
    28500 8.279948999403524
    29000 8.279948999237044
    29500 8.279948999123818
    30000 8.279948999046807
    30500 8.279948998994433
    31000 8.27994899895881
    31500 8.279948998934582
    32000 8.279948998918107
    32500 8.2799489989069
    33000 8.279948998899279
    33500 8.279948998894096
    34000 8.279948998890571
    34031 8.279948998890385
    34032 8.279948998890385 조기 종료!
    


```python
logits = X_test.dot(Theta)
Y_proba = sigmoid(logits)
y_predict = np.array([])
for i in range(len(Y_proba)):
  if Y_proba[i] >= 0.5:
    y_predict = np.append(y_predict, 1)
  else:
    y_predict = np.append(y_predict, 0)


accuracy_score = np.mean(y_predict == y_test)
accuracy_score
```




    0.9666666666666667




```python
#과제 2 : 구현한 로지스틱 회귀로 다중 분류 알고리즘을 구현
#일대다(OvR)방식 활용, 간당하게 a + b + c = 1 이라면 1 - a -b = c 를 활용한다는 뜻
#이걸 어떻게 구현하느냐 고민이 많았는데, "같은 코드를 두고 특성을 바꾸서 반복"같이 어렵게 하는게 아니라 단순하게 모델을 a, b 2개 만들면 해결되는 문제였음

X = iris["data"][:, (2, 3)]
y = iris["target"]
#모델 2개 만들기 위해서 판단모델 2개
y0 = (iris["target"] == 0).astype(np.int) #setosa 판단 모델 데이터셋
y2 = (iris["target"] == 2).astype(np.int) #virginica 판단 모델 데이터셋

X_with_bias = np.c_[np.ones([len(X), 1]), X]

np.random.seed(2042)

test_ratio = 0.2                                     
validation_ratio = 0.2                              
total_size = len(X_with_bias)                           

test_size = int(total_size * test_ratio)                
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size 

rnd_indices = np.random.permutation(total_size)


#마찬가지로 데이터셋 2개 준비
X_train = X_with_bias[rnd_indices[:train_size]] 
y_train = y[rnd_indices[:train_size]]
y_train0 = y0[rnd_indices[:train_size]] #setosa라벨
y_train2 = y2[rnd_indices[:train_size]] #virginica라벨

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
y_valid0 = y0[rnd_indices[train_size:-test_size]] #setosa라벨
y_valid2 = y2[rnd_indices[train_size:-test_size]] #virginica라벨

X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]

n_inputs = X_train.shape[1]
Theta0 = np.random.randn(n_inputs) #setosa세타값
Theta2 = np.random.randn(n_inputs) #virginica세타값
```


```python
#setosa 로지스틱 회귀모델

eta = 0.1 
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1           
best_loss0 = np.infty  

Theta0 = np.random.randn(n_inputs)

for iteration in range(n_iterations):

    logits0 = X_train.dot(Theta0)
    Y_proba0 = sigmoid(logits0)
    error = Y_proba0 - y_train0
    gradients0 = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1]), alpha * Theta0[1:]]
    Theta0 = Theta0 - eta * gradients0


    logits0 = X_valid.dot(Theta0)
    Y_proba0 = sigmoid(logits0)
    xentropy_loss0 = -np.mean(np.sum((y_valid0*np.log(Y_proba0 + epsilon) + (1-y_valid0)*np.log(1 - Y_proba0 + epsilon))))
    l2_loss0 = 1/2 * np.sum(np.square(Theta0[1:]))
    loss0 = xentropy_loss0 + alpha * l2_loss0
    

    if iteration % 500 == 0:
        print(iteration, loss0)
        

    if loss0 < best_loss0:
        best_loss0 = loss0
    else:                                     
        print(iteration - 1, best_loss0)       
        print(iteration, loss0, "조기 종료!")
        break
```

    0 20.721079478740872
    500 4.232989966878794
    1000 3.7863366883472254
    1500 3.681623866071943
    2000 3.6498101157987093
    2500 3.6391972424101517
    3000 3.635536957073207
    3500 3.6342596183187483
    4000 3.633812012781968
    4500 3.633654934248103
    5000 3.633599782375776
    


```python
eta = 0.1 
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1            
best_loss2 = np.infty   

Theta2 = np.random.randn(n_inputs)  

for iteration in range(n_iterations):

    logits2 = X_train.dot(Theta2)
    Y_proba2 = sigmoid(logits2)
    error = Y_proba2 - y_train2
    gradients2 = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1]), alpha * Theta2[1:]]
    Theta2 = Theta2 - eta * gradients2


    logits2 = X_valid.dot(Theta2)
    Y_proba2 = sigmoid(logits2)
    xentropy_loss2 = -np.mean(np.sum((y_valid2*np.log(Y_proba2 + epsilon) + (1-y_valid2)*np.log(1 - Y_proba2 + epsilon))))
    l2_loss2 = 1/2 * np.sum(np.square(Theta1[1:]))
    loss2 = xentropy_loss2 + alpha * l2_loss2

    
    if iteration % 500 == 0:
        print(iteration, loss2)
        

    if loss2 < best_loss2:
        best_loss2 = loss2
    else:                                     
        print(iteration - 1, best_loss2)       
        print(iteration, loss2, "조기 종료!")
        break
```

    0 58.87581646344981
    500 10.852770909188017
    1000 9.663973824207835
    1500 9.168070319573935
    2000 8.905642100299296
    2500 8.751390130844042
    3000 8.655545750178835
    3500 8.594031611586963
    4000 8.553753295473083
    4500 8.527040603472468
    5000 8.509176296582066
    


```python
#이제 1 - a - b 를 할 차례이다.
logits = X_test.dot(Theta0) #setosa확률값 추정  
setosa_proba = sigmoid(logits)

logits2 = X_test.dot(Theta2) #virginica확률값 추정 
virginica_proba = sigmoid(logits2)

y_predict = np.array([])
for i in range(len(Y_proba0)):
  prob_list = [[setosa_proba[i], 0], [1-setosa_proba[i]-virginica_proba[i], 1], [virginica_proba[i], 2]]
  prob_list.sort(reverse=True) #가장 높은 확률이 가장 앞으로 오게끔 정렬
  y_predict = np.append(y_predict, prob_list[0][1]) #가장 확률이 높았던 것을 예측값으로 결정

accuracy_score = np.mean(y_predict == y_test)
accuracy_score
```




    0.9333333333333333




```python

```
