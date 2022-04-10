# %%
import numpy as np 
from processData import data
from MLP import * 

np.random.seed(90)

epochs = 15
batch_size = 256 
lr = 0.1
decay_rate = 0.5
hidden_dim = 256
alpha = 0.1

X_train, Y_train, X_valid, Y_valid, X_test, Y_test, classes_num, input_dim = data()

model = MLP(input_dim, hidden_dim, classes_num)
model.optimize(X_train, Y_train, X_test, Y_test, epochs, batch_size, lr, decay_rate, alpha, print_log=True)

save_model(model)

# test evaluation 
# precision = model.precision(X_test, Y_test)
# print('Precision of test set: ', precision)

