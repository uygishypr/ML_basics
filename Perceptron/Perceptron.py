# %%
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import plotly
pd.options.plotting.backend = "plotly"
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# %%
#----------------Load Dataset ---------------------------
train_data = np.load('MNIST_data/train_data.npy')
train_labels = np.load('MNIST_data/train_labels.npy')

## Load the testing set
test_data = np.load('MNIST_data/test_data.npy')
test_labels = np.load('MNIST_data/test_labels.npy')

# ---------------Scale dataset-----------------------------
train_data_scaled = train_data/255
test_data_scaled = test_data/255

# Initialize one weight for each input pixel
weights = np.random.randint(-5, 6, 784)/10
bias = 1
#%%
def update(prior_results, perceptron_w, X, y, step_size):
    targets = [0,1,2,3,4,5,6,7,8,9]

    for i, target in enumerate(targets):
        if prior_results[i] > 0:
            output = 1
            if target != y:
                tk  = 0
                #perceptron0.weights = perceptron0.weights + step_size * (tk - output) * train_data[instance]
                perceptron_w.iloc[:,i] = perceptron_w.iloc[:,i] + step_size * (tk - output) * X
                
        else:
        # if activation is less than 0
            output = 0
            if target == y:
                tk = 1
                perceptron_w.iloc[:,i] = perceptron_w.iloc[:,i] + step_size * (tk - output) * X
    
    return perceptron_w

def predict(updated_weights, X_train, y_train, X_test, y_test):
    result_test = np.zeros((1000,10))
    result_train = np.zeros((7500,10))
    
    for i in range(X_test.shape[0]):
        result_test[i] = np.dot(X_test[i,:], updated_weights)
    
    for i in range(X_train.shape[0]):
        result_train[i] = np.dot(X_train[i,:], updated_weights)
        
    
    predicted_labels_test = result_test.argmax(axis=1)
    count_true_test = np.sum(predicted_labels_test == y_test)
    accuracy_test = count_true_test/(y_test.shape[0])
    
 
    predicted_labels_train = result_train.argmax(axis=1)
    count_true_train = np.sum(predicted_labels_train == y_train)
    accuracy_train = count_true_train/(y_train.shape[0])
    
    return accuracy_test, accuracy_train, predicted_labels_test
          
    
def train_perceptron(perceptron_weights, train_data, train_labels, test_data, test_labels, acc_history, step_size):
        
        # iterate thru 7500 train images
        for i in range(len(train_data)):
            prior_results = np.dot(train_data[i], perceptron_weights) + bias
            updated_perceptron_weights = update(prior_results, perceptron_weights, train_data[i], train_labels[i], step_size)
        
        acc_test, acc_train, predicted_labels_test = predict(updated_perceptron_weights, train_data, train_labels, test_data, test_labels)
        acc_history = acc_history.append({"accuracy_test": acc_test, "accuracy_train": acc_train}, ignore_index=True)
        return acc_history, predicted_labels_test
            

# %%
epochs = 100
perceptron_weights = pd.DataFrame(np.random.randint(-5, 6, size=(784, 10))/10, columns=["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"])
acc_history = pd.DataFrame({"accuracy_test": [], "accuracy_train": []})

for epoch_no in range(epochs):
    acc_history, predicted_labels_test = train_perceptron(perceptron_weights, train_data_scaled, train_labels, test_data_scaled, test_labels, acc_history, step_size=1.0)




cm = confusion_matrix(test_labels,predicted_labels_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0,1,2,3,4,5,6,7,8,9])

disp.plot()
plt.show()

acc_history.plot()

                              

# %%
# Define a function that displays a digit given its vector representation
def show_digit(x):
    plt.axis('off')
    plt.imshow(x.reshape((28,28)), cmap=plt.cm.gray)
    plt.show()
    return

## Define a function that takes an index into a particular data set ("train" or "test") and displays that image.
def vis_image(index, dataset="train"):
    if(dataset=="train"): 
        show_digit(train_data[index,:])
        label = train_labels[index]
    else:
        show_digit(test_data[index,:])
        label = test_labels[index]
    print("Label " + str(label))
    return


## Now view the first data point in the test set
vis_image(0, "test")

