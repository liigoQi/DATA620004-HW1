# MLP for MNIST
Yifan Qi

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Download Trained Model](https://drive.google.com/file/d/18BsUgZH_7KnEQq-Qi-wwPO9qdiSRCYDH/view?usp=sharing)


**Parameter searching:**
```
python3 paramSearch.py
```
The best combination of learning rate (0.1, 0.05, 0.01), dimension of hidden layer (64, 144, 256) and the parameter of L2 regularization (1, 0.1, 0.01) is found. The result of all combinations of parameters is saved in *search_parameters.csv*.

**Train with the best hyper-parameters:**
```
python3 train.py
```
Before training, data should be downloaded and put in *data* folder. After training, the trained model is saved as *model.pkl*.

**Test:**
```
python3 test.py
```
*model.pkl* is loaded and used to test. The model precisions of all digital numbers are printed. 