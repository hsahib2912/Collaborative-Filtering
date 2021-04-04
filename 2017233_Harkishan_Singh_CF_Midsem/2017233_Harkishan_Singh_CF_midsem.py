import numpy as np
import  numpy.linalg as npla
import pandas as pd


#This function creates R matrix ==> masking matrix
def get_r(train):
    R = []
    for i in range(len(train)):
        tmp = []
        for j in range(len(train[0])):
            if(train[i][j] == 0):
                tmp.append(0)
            else:
                tmp.append(1)
        R.append(tmp)
    return np.array(R)

#This function initializes X0 randomly
def init_x0(train):
    x0 = np.random.rand(train.shape[0],train.shape[1])
    return (x0)

#This is the main algorithm 
def nuclear_norm_minimization(train,lamda):
    X0 = init_x0(train)
    Y = train
    R = get_r(train)
    for i in range(50):           #You can change the number of iterations here. I have kept it to 10
        print("iteration = ",i)
        B = X0 + Y - np.multiply(R,X0)
        u,s,v = npla.svd(B)
        new_s = []
        for i in s:
            a = max(0,abs(i)-(lamda/2))
            if(i<0):
                a = -a
            new_s.append(a)
        sigma = []
        for i in range(len(u[0])):
            tmp = []
            for j in range(len(v)):
                if(i==j and i<len(new_s)):
                    tmp.append(new_s[i])
                else:
                    tmp.append(0)
            sigma.append(tmp)
        
        sigma = np.array(sigma)   
        X0 = np.dot(np.dot(u,sigma),v)
    return X0


#This function computes the mean absolute error
def mean_abs_err(test,pred):
    mae = 0
    common = 0
    for i in range(len(test)):
        for j in range(len(test[0])):
            if(test[i][j]!=0):
                common+=1
                mae+=abs(test[i][j]-pred[i][j])
    return mae/common

lamda = 10      #lamda value can be changed here
df = pd.read_csv('user_movie_rating.csv',header = None,index_col = False)
A = np.array(df)
X0 = nuclear_norm_minimization(A,lamda)
avg = 0

for i in range(5):
    start = int((i) * len(A)/5)
    end = int((i+1) * len(A)/5)
    print("Fold = ",i+1)
    mae = mean_abs_err(A[start:end],X0[start:end])
    avg+=mae
    print("Mean Absolute Error = ",mae)
print("Average MAE = ",avg/5)    