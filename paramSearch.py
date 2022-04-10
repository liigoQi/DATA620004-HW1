import numpy as np 
from MLP import MLP 
from itertools import product
from processData import data 
from tqdm import tqdm 

np.random.seed(90)

X_train, Y_train, X_valid, Y_valid, X_test, Y_test, classes_num, input_dim = data()

epochs = 12
batch_size = 256 
decay_rate = 0.5

lrs = [0.1, 0.05, 0.01]
hidden_dims = [64, 144, 256]
alphas = [1, 0.1, 0.01]

possible_params = dict(lrs=lrs, hidden_dims=hidden_dims, alphas=alphas)

def search_param(possible_params, log_file_name):
    result = dict()

    f = open(log_file_name, 'w')
    f.write('lr,hidden_dim,alpha,training_loss,training_acc,validation_loss,validation_acc\n')

    params = product(*possible_params.values())
    for param in tqdm(params):
        lr, hidden_dim, alpha = param 
        print()
        print('begin:', param)
        print()
        
        model = MLP(input_dim, hidden_dim, classes_num)
        model.optimize_train_only(X_train, Y_train, epochs, batch_size, lr, decay_rate, alpha, print_log=True)
        
        # use validation accuracy to choose the best combination of parameters 
        acc_valid, loss_valid = model.evaluate(X_valid, Y_valid)
        result[param] = acc_valid
        acc_train, loss_train = model.evaluate(X_train, Y_train)
        
        log = str(lr) + ',' + str(hidden_dim) + ',' + str(alpha) + ',' + str(loss_train) + ',' + str(acc_train) + ',' + str(loss_valid) + ',' + str(acc_valid) + '\n'
        f.write(log)
    
    best_param = max(result, key=result.get)
    print(best_param)
    final_log = 'best parameters: lr={}, hidden_dim={}, alpha={}'.format(*best_param)
    f.write(final_log)
    print(final_log)
    f.close()

search_param(possible_params, 'search_parameters.csv')