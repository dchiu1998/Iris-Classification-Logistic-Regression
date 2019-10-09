from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split


def normalize(X): 
    ''' 
    (Matrix) -> Matrix
    Given matrix M, normalize and return it.
    '''
    mins = np.min(X, axis = 0) 
    maxs = np.max(X, axis = 0) 
    diff = maxs - mins 
    norm_X = 1 - ((maxs - X)/diff) 
    return norm_X 

def show_plot(X,y,w):
    '''
    (Matrix, vector, vector) -> NoneType
    Plot the decision boundary using 2 features.
    '''
    # Get columns of features wanted (we'll plot feature 2 as independent
    # and feature 1 as dependent in this case)
    axis1 = X[:,2]
    axis2 = X[:,1]
    # Turn them into row vectors
    axis1 = axis1.reshape(1,-1)
    axis2 = axis2.reshape(1,-1)
    # Get all observations where y is 1
    axis1_1 = axis1[np.where(y==1)]
    axis2_1 = axis2[np.where(y==1)]
    # Get all observations where y is 0
    axis1_0 = axis1[np.where(y==0)]
    axis2_0 = axis2[np.where(y==0)]
    # plot all the points
    plot.scatter(axis1_0,axis2_0, c='b', label='y = 0') 
    plot.scatter(axis1_1, axis2_1, c='r', label='y = 1')     

    x1_plot = np.arange(0,1,0.1)
    # x2_plot given by rearranging equation w0 + w1x1 + w2x2 = 0 for x1
    # we get x1 = (-w0 - w2x2)/w1
    x2_plot = (-w[0,0] - w[0,2]*x1_plot)/w[0,1]
    # plot the decision boundary
    plot.plot(x1_plot,x2_plot,label = "Decision Boundary")
    plot.xlabel('Iris Feature 2')
    plot.ylabel('Iris Feature 1')
    plot.legend()
    plot.show()

def get_predictions(X,w):
    '''
    (Matrix, vector) -> vector
    Given a matrix of features, X, and vector of weights, w, predict
    the classification for each N observations of data.
    REQ: X has dimensions N x P
    REQ: w has dimensions 1 x P
    '''
    w = w.reshape(-1,1)
    # get the probability of outcome for each observation
    predictions = get_sigmoid(X,w)
    # classify each prediction
    for i in range(0, predictions.shape[1]):
        # if chance is >= 0.5, then classify as 1
        if (predictions[0,i] >= 0.5):
            predictions[0,i] = 1
        # otherwise classify as 0
        else:
            predictions[0,i] = 0
    #print(predictions)
    return predictions

def get_sigmoid(X,w):
    '''
    (Matrix, vector) -> vector
    Given feature matrix, X, and weight vector, w, return their logistic
    sigmoid function.
    REQ: X has dimensions N x P
    REQ: w has dimensions 1 x P
    '''
    w = w.reshape(-1,1)
    z = np.dot(X,w)
    return (1.0/(1+np.exp(-z))).T

def get_loss(X,y,w):
    '''
    (Matrix, vector, vector) -> float
    Given feature matrix, X, output vector, y, and weight vector, w,
    return their log-likelihood function.
    REQ: X has dimensions N x P+1
    REQ: y has dimensions 1 x N
    REQ: w has dimensions 1 x P
    '''
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    num_observations = len(y)
    # get the sigmoid functions representing the predictions
    predictions = get_sigmoid(X,w).T
    # get the 2 seperate cost functions
    # we do 1/N*(-y'*log(predictions) - (1-y)'*log(1-predictions))
    # to calculate cost; this is the vectorized version of the mse
    a1 = np.dot(y.T,np.log(predictions))
    a2 = np.dot((1-y).T, np.log(1-predictions))    
    mse = (-a1 - a2)/num_observations
    return mse

def get_gradient(X,y,w):
    '''
    (Matrix, vector, vector) -> vector
    Given the vector of features, X, vector of labels, y, and vector of
    weights, w, compute and return the gradient of the loss function.
    REQ: X has dimensions N x P
    REQ: y has dimensions 1 x N
    REQ: w has dimensions 1 x P
    '''
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    # get the vector of predictions
    predictions = get_sigmoid(X, w).T
    # turn y into a column vector
    # y.shape = (y.shape[0],1)
    # gradient = X'*(h(x)-y)
    grad = np.dot(X.T, predictions - y)
    return grad.T

def gradient_descent(X, y, w, alpha, tolerance):
    '''
    (Matrix, vector, vector, decimal, decimal) -> vector
    Perform gradient descent to find the optimal weights, w.
    '''
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    # get the loss function for the data
    loss = get_loss(X,y,w)
    loss_change = 1
    count = 0
    # continue gradient descent until change in loss is less than tolerance
    while(loss_change > tolerance):
        # update the weights
        w = w - (alpha * get_gradient(X,y,w).T)
        # compute the new change in loss
        old_loss = loss
        loss = get_loss(X,y,w)
        loss_change = old_loss - loss
        count = count + 1

    print(count, "iteration(s) for alpha level", alpha, "and tolerance of"
          , tolerance)
    return w.T

def get_efficiency(output,predictions):
    '''
    (vector,vector) -> NoneType
    Compute and print the F1 score, recall and precision.
    REQ: output has dimensions 1 x N
    REQ: predictions has dimensions 1 x N
    '''
    true_positive = 0
    false_negative = 0
    false_positive = 0
    # go through every element in vectors
    for i in range(0,yTest.shape[1]):
        if(output[0,i] == 1 and predictions[0,i]==1):
            true_positive+=1
        if(predictions[0,i]==0 and output[0,i]==1):
            false_negative+=1
        if(predictions[0,i]==1 and output[0,i]==0):
            false_positive+=1
    # compute the scores
    recall = true_positive/(true_positive+false_negative)
    precision = true_positive/(true_positive+false_positive)
    f1_score = 2*precision*recall/(precision+recall)
    print("Recall:",recall)
    print("Precision:",precision)
    print("F1 score:", f1_score)

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:100,:]
    y = iris.target[:100] # the labels
    # split the data 80/20 for training and testing data respectively
    XTrain, XTest, yTrain, yTest = train_test_split(X,y,test_size=0.2)

    # normalize the feature vectors
    XTrain = normalize(XTrain)
    XTest = normalize(XTest)

    # append a column of 1's to the feature matrix to account for bias terms
    ones = np.ones(XTrain.shape[0])
    ones = ones.reshape(-1,1)
    ones_test = np.ones(XTest.shape[0])
    ones_test = ones_test.reshape(-1,1)
    XTrain = np.hstack((ones,XTrain))
    XTest = np.hstack((ones_test,XTest))

    # turn outputs into row vectors
    yTrain = yTrain.reshape(1,-1)
    yTest = yTest.reshape(1,-1)

    # create a vector of 0's as intial weight vector
    w = np.matrix(np.zeros(XTrain.shape[1]))

    # find optimal weights
    alpha = 0.001
    tolerance = 0.9
    w = gradient_descent(XTrain,yTrain,w,alpha,tolerance)

    # get the predictions on the test data
    prediction_test = get_predictions(XTest,w)

    # plot the decision boundary
    show_plot(XTest,yTest,w)

    # get the accuracy scores
    get_efficiency(yTest,prediction_test)