import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
import pandas as pd
from scipy import ndimage
from spamutil import *
def L_layer_model(X, Y, layers_dims, learning_rate = 0.009, num_iterations = 100, print_cost=False):#lr was 0.009
  
    np.random.seed(1)
    costs = []            
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten
test_x = test_x_flatten
layers_dims = [57, 20, 7, 5, 5, 1] #  5-layer model

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
new = np.array([[0,0,0.23,0,0,0,0.23,0,0,0.95,0,0.47,0,0.23,0,0.23,0.95,0,2.38,0,1.9,0,0,0.47,0,0,0,0,0,0,0,0,0,0,0,0.23,0.23,0,0,0,0,0,0,0,0,0,0,0,0,0.123,0,0.197,0,0.024,5.038,280,519]]).reshape((57,1))
pr = predict(new, [1], parameters)
if(np.squeeze(pr) == 1):
    print("Given email is a Spam.")
else:
    print("Given email is not a Spam.")
"""
my_image = "cats.jpg" # change this to the name of your image file
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
"""
