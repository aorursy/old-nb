#!/usr/bin/env python
# coding: utf-8



# Though the accuracy is less but written this program just for sake of writing the single 
# hidden layer neural network from scratch




import cv2
import numpy as np 
import os 
from random import shuffle
from tqdm import tqdm
import scipy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




train_directory = "/Users/Mak/Desktop/dogvscat/train"
test_directory = "/Users/Mak/Desktop/dogvscat/test"




def label_img(img):
    name = img.split('.')[0]
    if name == "dog":
        return 1
    elif name == "cat":
        return 0




def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(train_directory)):
        label = label_img(img)
        path = os.path.join(train_directory,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32,32))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data




def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(test_directory)):
        img_num = img.split('.')[0]
        path = os.path.join(test_directory,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32,32))
        testing_data.append([np.array(img),img_num])
    shuffle(testing_data)
    np.save('testing_data.npy', testing_data)
    return testing_data




# will only run once if training/test.npy is not created 
df_train= create_train_data()
df_test = create_test_data()




# df_test  = np.load("testing_data.npy") # given by source [test]
# df_train = np.load("train_data.npy")
# # only dealing with training data 




df_train_feature = np.array([i[0] for i in df_train])
df_train_label   = [i[1] for i in df_train] #gives list 
df_train_label = np.asarray(df_train_label) #gives a array  with shape (123121...,)
# .reshape(-1,32,32)
print df_train_feature.shape
print df_train_label.shape
# (25000, 32, 32)
# (25000,)




df_train_label = df_train_label.reshape(df_train_label.shape[0], 1)
df_train_label.shape




df_train_feature = df_train_feature.reshape(df_train_feature.shape[0],-1) # flatten the feature matrix 
print df_train_feature.shape
print df_train_label.shape




from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_train_feature, df_train_label, test_size=0.3)




x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T




print x_train.shape
print y_train.shape
print x_test.shape 
print y_test.shape
# Data has been created 
# 17500 images are in training data and 7500 images in testing data 




x_train = x_train/255
x_test = x_test/255




def sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig 




def layer_size(x,y):
    nx = x.shape[0]
    nhl = 4
    ny = y.shape[0]
    return nx , nhl , ny




def initialize_parameter(nx,nhl,ny,factor):
    w1 = np.random.randn(nhl,nx)*factor
    b1 = np.zeros((nhl,1))
    w2 = np.random.randn(ny,nhl)*factor
    b2 = np.zeros((ny,1))
    parameters = {
        "w1":w1,
        "w2":w2,
        "b1":b1,
        "b2":b2   
    }
    return parameters  




def forward_prop(x, parameters):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    z1 = np.dot(w1, x)+b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2,a1)+b2
    a2 = sigmoid(z2)
    cache = {"a1":a1,
             "a2":a2,
             "z1":z1,
             "z2":z2
            }
    return a2 , cache




def costfunction(y, a2, parameters):
    m = y.shape[1]
    cost = (1/(-m))*(np.sum(np.multiply(np.log(a2),y) + np.multiply(np.log(1-a2),1-y)))
    cost = np.squeeze(cost)
    return cost 




def backward_propagation(parameters, cache , x, y):
    m = x.shape[1]
    w1=parameters["w1"]
    w2=parameters["w2"]
    a1=cache["a1"]
    a2=cache["a2"]
    dz2 = (a2-y)
    dw2 = (1.0/m)*np.dot(dz2,a1.T)
    db2 = (1.0/m)*np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T,dz2)*(1 - np.power(a1, 2))
#     print dz1.shape,"helllllooo", x.T.shape
#     print pd.DataFrame(dz1).describe()
#     df__ = x.T
#     print df__.describe()
    dw1 = np.dot(dz1,x.T)/m*1.0
#     print dw1,"identity"
    db1 = (1.0/m)*np.sum(dz1, axis=1, keepdims=True)
    grads = {"dw1": dw1,
             "db1": db1,
             "dw2": dw2,
             "db2": db2}
    return grads




def update_parameters(parameters, grads, learning_rate):
    dw1 = grads["dw1"]
    dw2 = grads["dw2"]
    db1 = grads["db1"]
    db2 = grads["db2"]
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    w1 = w1 - learning_rate*dw1
    w2 = w2 - learning_rate*dw2
    b1 = b1 - learning_rate*db1
    b2 = b2 - learning_rate*db2
    parameters = {
             "w1":w1,
             "w2":w2,
             "b1":b1,
             "b2":b2
            }
    return parameters




def nn_model(x, y, nhl, iterations = 1000, learning_rate = 2,  print_cost = False):
    nx= x.shape[0]
    ny= y.shape[0]
    parameters = initialize_parameter(nx,nhl,ny,0.1)
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    costs=[]
    iteration =[]
    for i in range(iterations):
        a2 , param = forward_prop(x , parameters)
        cost = costfunction(y, a2, parameters)
        if i % 100 == 0:
            costs.append(cost)
            iteration.append(i)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        grads =backward_propagation(parameters,param, x, y)
        parameters =update_parameters(parameters, grads, learning_rate )
    plt.plot(iteration, costs)
    plt.xlabel('iter_num')
    plt.ylabel('COST')
    plt.title('LEARNING_PERIOD')
    plt.grid(True)
    plt.show()
    print cost 
    return parameters 




parameter = nn_model(x_train, y_train, nhl = 4 , iterations = 1000, print_cost=True)




def predict1(parameters, X):
    a2, cache = forward_prop(X, parameters)
    print a2.shape
    y_predict = np.zeros((1,a2.shape[1]))
    for i in range(a2.shape[1]):
        if  a2[0,i] <= 0.5:
            y_predict[0][i]=0
        elif a2[0,i]>0.5:
            y_predict[0][i]=1
    return y_predict




prediction_test  = predict1(parameter, x_test)
print prediction_test
prediction_train  = predict1(parameter, x_train)
print prediction_train
print("test accuracy: {} %".format(100 - (np.mean(np.abs(prediction_test - y_test)) * 100)))
print("train accuracy: {} %".format(100 - (np.mean(np.abs(prediction_train - y_train)) * 100)))






