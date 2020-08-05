# -*- coding: utf-8 -*-
"""forBoard.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WClsqR9WRPWnX-2lLv8emuQytb1fbRms

Download data
"""


import optim
import pickle
import sys
import os

if len(sys.argv) != 3:
    print("Error, require two number input. e.g. python3 client.py 1 1")
    sys.exit()
ini = int(sys.argv[1])
client = (sys.argv[2])   
csv_file = 'test.csv'
#'test.csv'
testSet = optim.myDataset(csv_file=csv_file)
csv_file = 'train'+client+'.csv'
#'train.csv'
trainSet = optim.myDataset(csv_file=csv_file)


weight_scale = 2e-2
hidden_dims = [100, 100]
num_layers = 3
bn_model = None


# initial gradient
f=open("weights\weight_bin.bin","rb")
load_w = {}
load_w['W1'] = pickle.load(f)
load_w['b1'] = pickle.load(f)
load_w['W2'] = pickle.load(f)
load_w['b2'] = pickle.load(f)
load_w['W3'] = pickle.load(f)
load_w['b3'] = pickle.load(f)
load_w['gamma1'] = pickle.load(f)
load_w['beta1'] = pickle.load(f)
load_w['gamma2'] = pickle.load(f)
load_w['beta2'] = pickle.load(f)

load_bn = None
f.close()
if ini == 1:
    if os.path.exists("history\clientTrainMSELoss"+client+".bin"):
        os.remove("history\clientTrainMSELoss"+client+".bin")
        print("client " + client+" clears train loss history")
    if os.path.exists("history\clientTestMSELoss"+client+".bin"):
        os.remove("history\clientTestMSELoss"+client+".bin")
        print("client " + client+" clears test loss history")
    load_bn = [{'mode': 'train'} for i in range(num_layers - 1)]
    #for i in range(num_layers - 1):
    #    load_bn[i]['running_mean'] = pickle.load(f)
    #    load_bn[i]['running_var'] = pickle.load(f)
else:
    f=open("weights\weight_bin"+client+".bin","rb")
    pickle.load(f)
    pickle.load(f)
    pickle.load(f)
    pickle.load(f)
    pickle.load(f)
    pickle.load(f)
    pickle.load(f)
    pickle.load(f)
    pickle.load(f)
    pickle.load(f)
    load_bn = [{'mode': 'train'} for i in range(num_layers - 1)]
    for i in range(num_layers - 1):
        load_bn[i]['running_mean'] = pickle.load(f)
        load_bn[i]['running_var'] = pickle.load(f)
    f.close()
    print("client " + client+" loads the local batch layer parameters")
bn_model = optim.FullyConnectedNet(hidden_dims, weight_scale=weight_scale, load_weights=load_w, load_bn=load_bn)
print("client " + client+" loads the global model")
trainLoss, grads = bn_model.loss(trainSet.pos, trainSet.RSSI)
f = open('weights\weight_bin'+client+'.bin','wb')
for para in bn_model.params:
    pickle.dump(bn_model.params[para], f)
for para in bn_model.bn_params:
    pickle.dump(para['running_mean'],f)
    pickle.dump(para['running_var'],f)
f.close()
print('client ' + client+' stores the model weight')

f = open('gradients\grads_bin'+client+'.bin','wb')
for grad in grads:
    pickle.dump(grads[grad], f)
f.close()
print('client ' + client+' obtains and stores full gradient')

f = open('history\clientTrainMSELoss'+client+'.bin','ab')
pickle.dump(trainLoss, f)
f.close()

testScores = bn_model.loss(testSet.pos)
testLoss, _ = optim.mse_loss(testScores, testSet.RSSI)
f = open('history\clientTestMSELoss'+client+'.bin','ab')
pickle.dump(testLoss, f)
f.close()

print('client ' + client+' records the train and test MSE loss')
