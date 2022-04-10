import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
from processData import data

np.random.seed(90)

def relu(z):
    return z * (z > 0)

def drelu(z):
    return (z > 0)

def softmax(z):
    tmp = z - np.max(z, axis=1, keepdims=True)
    return np.exp(tmp) / np.sum(np.exp(tmp), axis=1, keepdims=True)

def cross_entropy(Y_hat, Y):
    delta = 1e-7
    return -np.sum(Y * np.log(Y_hat + delta)) / len(Y)

def accuracy(Y_hat, Y):
    return np.sum(np.argmax(Y_hat, axis=1) == np.argmax(Y, axis=1)) / len(Y)

def load_model(model_name):
    with open(model_name, 'rb') as f:
        model = pickle.load(f)
    return model 

def save_model(model):
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)


class MLP(object):
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''
        two layer classifier: MLP with one hidden layer and one output layer
        
        input(a[0]) -> hidden layer(z[0]) -> relu(a[1]) -> output layer(z[1]) -> softmax(a[2])

        :input_dim: dimension of input feature 
        :hidden_dim: dimension of hidden layer
        :output_dim: dimension of output, i.e., the number of classes
        '''
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 

        self.reset_parameters()
        
        self.a = [None for _ in range(3)]
        self.z = [None for _ in range(2)]

        self.reset_log()
    
    def reset_parameters(self):
        self.W1 = np.random.normal(size=(self.hidden_dim, self.input_dim)) * np.sqrt(2 / self.input_dim)
        self.b1 = np.zeros((self.hidden_dim, 1))
        self.dW1 = None 
        self.db1 = None

        self.W2 = np.random.normal(size=(self.output_dim, self.hidden_dim)) * np.sqrt(2 / self.hidden_dim)
        self.b2 = np.zeros((self.output_dim, 1))
        self.dW2 = None 
        self.db2 = None 
    
    def reset_log(self):
        self.loss_train_list = []
        self.loss_test_list = []
        self.accuracy_test_list = []
    
    def forward(self, X):
        self.a[0] = X
        self.z[0] = np.dot(self.a[0], self.W1.T) + self.b1.T
        self.a[1] = relu(self.z[0])
        self.z[1] = np.dot(self.a[1], self.W2.T) + self.b2.T
        self.a[2] = softmax(self.z[1])
    
    def backpropagate(self, Y, rate, alpha):
        # batch_size: 256, hidden_dim: 128, output_dim: 10 
        dl2 = self.a[2] - Y  # 256*10
        # L2 regularization
        self.dW2 = np.dot(dl2.T, self.a[1]) + alpha * self.W2 # 10*128 
        self.db2 = np.sum(dl2.T, axis=1, keepdims=True) # 10*1
        
        dl1 = np.dot(dl2, self.W2) * drelu(self.z[0])   # 256*128
        # L2 regularization
        self.dW1 = np.dot(dl1.T, self.a[0]) + alpha * self.W1 # 128*784
        self.db1 = np.sum(dl1.T, axis=1, keepdims=True) # 128*1
        
        # update 
        self.W1 -= rate * self.dW1
        self.W2 -= rate * self.dW2 
        self.b1 -= rate * self.db1
        self.b2 -= rate * self.db2 

        # reset 
        self.dW1 = None 
        self.dW2 = None 
        self.db1 = None 
        self.db2 = None 
    
    def predict(self, X):
        Y_hat = np.dot(X, self.W1.T) + self.b1.T
        Y_hat = relu(Y_hat)
        Y_hat = np.dot(Y_hat, self.W2.T) + self.b2.T 
        Y_hat = softmax(Y_hat)
        return Y_hat 
    
    def get_batchs(self, total_num, batch_size):
        idx = np.random.permutation(total_num)
        return [idx[i:i+batch_size] for i in range(0, len(idx), batch_size)]

    def optimize_train_only(self, X_train, Y_train, epochs, batch_size, lr=0.2, decay_rate=1, alpha=0.1, print_log=False):
        for epoch in range(epochs):
            # mini-batch 
            batchs = self.get_batchs(len(X_train), batch_size)
            for i, batch in enumerate(batchs):
                X_batch, Y_batch = X_train[batch], Y_train[batch]
                self.forward(X_batch)
                self.backpropagate(Y_batch, lr / batch_size, alpha)
                # training evaluation 
                accuracy_train, loss_train = self.evaluate(X_train, Y_train)

            if print_log:
                print('epoch: {} | train loss: {:.4f} | accuracy: {:.4f}'.format(epoch+1, loss_train, accuracy_train))
           
            # learning rate decay 
            lr = 1 / (1 + decay_rate * epoch) * lr 

            # early stop
            if len(self.loss_train_list) >= 2 and abs(self.loss_train_list[-1] - self.loss_train_list[-2]) < 1e-3:
                break 

    def optimize(self, X_train, Y_train, X_test, Y_test, epochs, batch_size, lr=0.2, decay_rate=1, alpha=0.1, print_log=False):

        for epoch in range(epochs):
            # mini-batch 
            batchs = self.get_batchs(len(X_train), batch_size)
            for i, batch in enumerate(batchs):
                X_batch, Y_batch = X_train[batch], Y_train[batch]
                self.forward(X_batch)
                self.backpropagate(Y_batch, lr / batch_size, alpha)
                
                accuracy_train, loss_train = self.evaluate(X_batch, Y_batch)
                
                if print_log:
                    print('epoch: {} | batch: {} | train loss: {:.4f} | accuracy: {:.4f}'.format(epoch + 1, i + 1, loss_train, accuracy_train))
            
            # training evaluation 
            accuracy_train, loss_train = self.evaluate(X_train, Y_train)
            # test evaluation
            accuracy_test, loss_test = self.evaluate(X_test, Y_test)

            # record metric 
            self.accuracy_test_list.append(accuracy_test)
            self.loss_train_list.append(loss_train)
            self.loss_test_list.append(loss_test)

            # learning rate decay 
            lr = 1 / (1 + decay_rate * epoch) * lr 

            # early stop
            # if train loss does not change in 2 continuous epochs, stop training 
            if len(self.loss_train_list) >= 2 and abs(self.loss_train_list[-1] - self.loss_train_list[-2]) < 1e-3:
                self.plot_loss(epoch + 1)
                self.plot_test_acc(epoch + 1)
                self.visualize_weights(epoch + 1)
                break 

            if (epoch+1) % 5 == 0 and print_log:
                self.plot_loss(epoch + 1)
                self.plot_test_acc(epoch + 1)
                self.visualize_weights(epoch + 1)

    def plot_loss(self, epoch):
        '''
        visualize the loss line of training and test set, using self.loss_train_list & self.loss_test_list 
        '''
        epochs = list(range(1, len(self.loss_train_list) + 1))
        plt.plot(epochs, self.loss_train_list, '.-', label='Train Loss', color='b')
        plt.plot(epochs, self.loss_test_list, '.-', label='Test Loss', color='r')
        plt.legend(['Train Loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        image_name = 'epoch{}-loss.jpg'.format(epoch)
        plt.savefig(image_name)
        plt.close()
        #plt.show()

    def plot_test_acc(self, epoch):
        '''
        visualize the accuracy line of test set using self.accuracy_test_list 
        '''
        epochs = list(range(1, len(self.accuracy_test_list) + 1))
        plt.plot(epochs, self.accuracy_test_list, '.-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        image_name = 'epoch{}-test_acc.jpg'.format(epoch)
        plt.savefig(image_name)
        plt.close()
        #plt.show()

    def visualize_weights(self, epoch):
        '''
        visualize parameters of each layer
        '''
        # W1: hidden_dim(~=sqrt(hidden_dim)*sqrt(hidden_dim)) * input_dim(28*28)
        nrows, ncols = int(self.hidden_dim ** 0.5), int(self.hidden_dim ** 0.5)
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows, ncols))
        for k, ax in enumerate(axes.flatten()):
            ax.imshow(self.W1[k].reshape(28, 28), cmap='viridis')
            ax.axis('off')
        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.02, hspace=0.02)
        image_name = 'epoch{}-weights1.jpg'.format(epoch)
        plt.savefig(image_name)
        plt.close()
        #plt.show()
        
        # W2: output_dim(10) * hidden_dim(~=sqrt(hidden_dim)*sqrt(hidden_dim))
        nrows, ncols = 2, 5
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows, ncols))
        for k, ax in enumerate(axes.flatten()):
            ax.imshow(self.W2[k].reshape(int(self.hidden_dim ** 0.5), int(self.hidden_dim ** 0.5)), cmap='viridis')
            ax.axis('off')
        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.02, hspace=0.02)
        image_name = 'epoch{}-weights2.jpg'.format(epoch)
        plt.savefig(image_name)
        plt.close()
        #plt.show()

    def evaluate(self, X, Y):
        Y_hat = self.predict(X)
        acc = accuracy(Y_hat, Y)
        loss = cross_entropy(Y_hat, Y)
        return acc, loss 

    def precision(self, X, Y):
        '''
        compute precisions of all ten classes
        precision = TP / (TP + FP)
        '''
        Y_hat = self.predict(X)
        true_labels = np.argmax(Y, axis=1)
        pred_labels = np.argmax(Y_hat, axis=1)
        precision_dict = dict(zip(list(range(10)), [0 for _ in range(10)]))
        for k in range(10):
            TP = sum((true_labels == k) & (pred_labels == k))
            precision_dict[k] = TP / sum(pred_labels == k)
        return precision_dict
        
if __name__ == "__main__":

    epochs = 10
    batch_size = 256 
    lr = 0.01
    decay_rate = 0.5
    hidden_dim = 300
    alpha = 0.5

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test, classes_num, input_dim = data()

    model = MLP(input_dim, hidden_dim, classes_num)

    model.optimize(X_train, Y_train, X_test, Y_test, epochs, batch_size, lr, decay_rate, alpha, print_log=True)

    # test evaluation 
    precision = model.precision(X_test, Y_test)
    print('Precision of test set: ', precision)

