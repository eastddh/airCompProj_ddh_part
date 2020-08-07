import numpy as np
import pandas as pd
import optim
import pickle
import sys
import os

if len(sys.argv) != 5:
    print("Error, require 0/1 + iteration+ initial lr+decay as input. e.g. python3 server.py 1 1 1e-4 0.9")
    sys.exit()

weight_scale = 2e-2
hidden_dims = [100, 100]
num_layers = 3

ini = int(sys.argv[1])
epoch = int(sys.argv[2]) 
lr = float(sys.argv[3])
decay = float(sys.argv[4])
lr = lr* (decay** (epoch-1))
if epoch % 10 ==0:
    print("Iteration ", epoch," with lr ", lr)
# initial gradient
if ini == 1:
    bn_model = optim.FullyConnectedNet(hidden_dims, weight_scale=weight_scale)
    print("Server initializes global model weight")
    # write down the initial random model weight
    f = open('weights/weight_bin.bin','wb')
    optim.write_bin('weights/weight_bin.bin', bn_model.params)
    print('Server stores the model weight')
else:
    #########################################
    ###     Server reads model weights    ###
    #########################################
    #f=open("weights\weight_bin.bin","rb")
    load_w = optim.read_bin("weights/weight_bin.bin")
##    load_w['W1'] = pickle.load(f)
##    load_w['b1'] = pickle.load(f)
##    load_w['W2'] = pickle.load(f)
##    load_w['b2'] = pickle.load(f)
##    load_w['W3'] = pickle.load(f)
##    load_w['b3'] = pickle.load(f)
##    load_w['gamma1'] = pickle.load(f)
##    load_w['beta1'] = pickle.load(f)
##    load_w['gamma2'] = pickle.load(f)
##    load_w['beta2'] = pickle.load(f)
        
    load_bn = [{'mode': 'train'} for i in range(num_layers - 1)]
    #for i in range(num_layers - 1):
    #    load_bn[i]['running_mean'] = pickle.load(f)
    #    load_bn[i]['running_var'] = pickle.load(f)
    
    bn_model = optim.FullyConnectedNet(hidden_dims, weight_scale=weight_scale, load_weights=load_w, load_bn=load_bn)
    #print("Server finished Loading model weights")

    #########################################
    ###     Server reads aggregated       ###
    ###     gradients.                    ###
    #########################################
    load_grads = optim.read_bin("gradients/grads_bin.bin")
##    f=open("gradients\grads_bin.bin","rb")
##    load_grads = {}
##    load_grads['W3'] = pickle.load(f)
##    load_grads['b3'] = pickle.load(f)
##    load_grads['gamma2'] = pickle.load(f)
##    load_grads['beta2'] = pickle.load(f)
##    load_grads['W2'] = pickle.load(f)
##    load_grads['b2'] = pickle.load(f)
##    load_grads['gamma1'] = pickle.load(f)
##    load_grads['beta1'] = pickle.load(f)
##    load_grads['W1'] = pickle.load(f)
##    load_grads['b1'] = pickle.load(f)
##    f.close()
    optim_config={'learning_rate': lr,
                      'momentum': 0.9,
                    }
    optim_configs={}
    for p in bn_model.params:
        d = {k: v for k, v in optim_config.items()}
        optim_configs[p] = d
    #lr_decay=0.95

    # update
    for p, w in bn_model.params.items():
        dw = load_grads[p]
        config = optim_configs[p]
        next_w, next_config = optim.sgd_momentum(w, dw, config)
        bn_model.params[p] = next_w
        optim_configs[p] = next_config
    #print("Server updates the model weights")
    optim.write_bin('weights/weight_bin.bin', bn_model.params)
    #f = open('weights\weight_bin.bin','wb')
    #for para in bn_model.params:
      #print(para)
    #  pickle.dump(bn_model.params[para], f)
    #for para in bn_model.bn_params:
    #    pickle.dump(para['running_mean'],f)
    #    pickle.dump(para['running_var'],f)
    #f.close()
    #print('Server stores the model weight')



