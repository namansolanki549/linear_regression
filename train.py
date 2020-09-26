import numpy as np
import csv
#from scan_utils import scan_X, scan_Y, scan_W

def import_data():
    X = np.genfromtxt("train_X_lr.csv", dtype=np.float64, delimiter=',', skip_header=1)
    Y = np.genfromtxt("train_Y_lr.csv", dtype=np.float64, delimiter=',')
    return X, Y

def compute_cost(X, Y, W):
    #TODO Complete the function implementation. Read the Question text for details
    Y_pred = np.dot(X, W)
    return(np.sum(np.square(Y_pred - Y)))/(2*len(X))
    
def compute_gradient_of_cost_function(X, Y, W):
    #TODO Complete the function implementation. Read the Question text for details
    Y_pred = np.dot(X, W)
    return((np.dot((Y_pred - Y).T, X))/len(X)).T

def optimize_weights_using_gradient_descent(X, Y, W, num_iterations, learning_rate):
    #TODO Complete the function implementation. Read the Question text for details
    previous_iter_cost = 0
    iter_no = 0
    for i in range(num_iterations):
        dW = compute_gradient_of_cost_function(X, Y, W)
        W = W - (learning_rate * dW)
        cost = compute_cost(X, Y, W)
        print(i, cost)
    return W
def train_model(X, Y):
    X = np.insert(X, 0, 1, axis = 1)
    Y = Y.reshape(len(X), 1)
    W = np.zeros((X.shape[1], 1))
    W = optimize_weights_using_gradient_descent(X, Y, W, 57715170, 0.0002)
    return W

def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()

if __name__ == '__main__':
    X, Y = import_data()
    weights = train_model(X, Y)
    save_model(weights, "WEIGHTS_FILE.csv")
    #print(X[:3])