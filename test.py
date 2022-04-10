# %%
from MLP import * 
from processData import data 

def print_precision(precision):
    for k in precision.keys():
        print('Precision of {}: {:.4f}'.format(k, precision[k]))

X_train, Y_train, X_valid, Y_valid, X_test, Y_test, classes_num, input_dim = data()
model = load_model('model.pkl')

# %%
# test evaluation 
precision = model.precision(X_test, Y_test)
print_precision(precision)

# %%
