import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 

    classes = np.unique(y)
    d = X.shape[1]
    k = len(classes)
    means = np.zeros((d, k))

    for i, c in enumerate(classes):
        means[:, i] = np.mean(X[y.flatten() == c], axis=0)

    covmat = np.cov(X, rowvar=False)
    return means,covmat


def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    classes = np.unique(y)
    d = X.shape[1]
    k = len(classes)
    means = np.zeros((d, k))
    covmats = []

    for i, c in enumerate(classes):
        class_data = X[y.flatten() == c]
        means[:, i] = np.mean(class_data, axis=0)
        covmats.append(np.cov(class_data, rowvar=False))

    return means,covmats


def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    inv_covmat = inv(covmat)
    scores = []

    for i in range(means.shape[1]):
        diff = Xtest - means[:, i].T
        scores.append(-0.5 * np.sum(diff @ inv_covmat * diff, axis=1))

    scores = np.array(scores).T
    ypred = np.argmax(scores, axis=1) + 1
    acc = np.mean(ypred.flatten() == ytest.flatten())
    ypred = ypred.reshape(-1, 1)
    return acc,ypred


def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    scores = []

    for i in range(means.shape[1]):
        diff = Xtest - means[:, i].T
        inv_covmat = inv(covmats[i])
        logdet_covmat = np.log(det(covmats[i]))
        scores.append(-0.5 * (np.sum(diff @ inv_covmat * diff, axis=1) + logdet_covmat))

    scores = np.array(scores).T
    ypred = np.argmax(scores, axis=1) + 1
    acc = np.mean(ypred.flatten() == ytest.flatten())
    ypred = ypred.reshape(-1, 1)
    return acc,ypred


def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD     
    w = inv(X.T @ X) @ X.T @ y                                              
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    d = X.shape[1]
    I = np.eye(d)
    w = inv(X.T @ X + lambd * I) @ X.T @ y                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    ypred = Xtest @ w
    mse = np.mean((ytest - ypred) ** 2)
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD      
    w = w.reshape(-1, 1)
    error = 0.5 * np.sum((y - X @ w) ** 2) + 0.5 * lambd * np.sum(w ** 2)
    error_grad = -X.T @ (y - X @ w) + lambd * w
    error_grad = error_grad.flatten() 

    # print('SHAPE', error_grad.shape, error)                                      
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    N = x.shape[0]
    Xp = np.ones((N, p + 1))
    for i in range(1, p + 1):
        Xp[:, i] = x.flatten() ** i
    return Xp

# Main script

# Problem 1
# load the sample data  
print('Problem 1 starts')                                                               
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
print('Problem 2 starts')                                                               
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

error_train = testOLERegression(w, X, y)
error_train_i = testOLERegression(w_i, X_i, y)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

print('Train MSE without intercept '+str(error_train))
print('Train MSE with intercept '+str(error_train_i))

# Problem 3
print('Problem 3 starts')                                                               
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
opt_lambd = 0
opt_w = None
min_error = float('inf')
min_train_error = float('inf')

for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    
    if min_error > mses3[i]:
        min_error = mses3[i]
        opt_lambd = lambd
        opt_w = w_l
        min_train_error = mses3_train[i]
    i = i + 1

print('OPTIMAL LAMBDA', opt_lambd)
print('Comparing ole & ridge regression weights')
print('Difference b/w weights', np.sum(w_i - opt_w))
print('Train & Test error for optimal lambda', min_train_error, min_error)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
print('Problem 4 starts')                                                               
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1]))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

opt_iter = np.argmin(mses4)
print('Optimal training error', mses4_train[opt_iter])
print('Optimal test error', mses4[opt_iter])
print('Optimal lambda', lambdas[opt_iter])

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
print('Problem 5 starts')                                                               
pmax = 7
lambda_opt = opt_lambd # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
p1_zero_lambda = 0
p2_opt_lambda = 0
min_mses_zero_lambda = float('inf')
min_mses_opt_lambda = float('inf')

for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

    if min_mses_zero_lambda > mses5[p, 0]:
        min_mses_zero_lambda = mses5[p, 0]
        p1_zero_lambda = p

    if min_mses_opt_lambda > mses5[p, 1]:
        min_mses_opt_lambda = mses5[p, 1]
        p1_opt_lambda = p

print('Optimal p for 0 lambda', p1_zero_lambda)
print('Optimal p for optimal lambda', p1_opt_lambda)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
