import numpy as np
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
#ini = 1 then server initializes global model weight and broadcast to clients
epoch = int(sys.argv[2]) #current epoch  
lr = float(sys.argv[3]) #initial learning rate
decay = float(sys.argv[4]) # learning rate decay 
c_lr = lr* (decay** (epoch-1)) # current learning rate
if epoch % 10 ==0:
    print("Iteration ", epoch," with lr ", c_lr)
# initial model weight
if ini == 1:
    if os.path.exists("server_info/velocity.bin"):
        os.remove("server_info/velocity.bin")
        print("server initializes update config")
    bn_model = optim.FullyConnectedNet(hidden_dims, weight_scale=weight_scale)
    print("Server initializes global model weight")
    # write down the initial random model weight
    optim.write_bin('weights/weight_bin.bin', bn_model.params)
    
    f = open("server_info/velocity.bin",'wb')
    for para in bn_model.params:
        pickle.dump(para, f)
        pickle.dump(np.zeros_like(bn_model.params[para]), f)
    f.close()
    print('Server stores the model weight')
else:
    #########################################
    ###     Server reads model weights    ###
    #########################################
    load_w = optim.read_bin("weights/weight_bin.bin") # read previous w
    load_bn = [{'mode': 'train'} for i in range(num_layers - 1)] 
    #bn_model = optim.FullyConnectedNet(hidden_dims, weight_scale=weight_scale, load_weights=load_w, load_bn=load_bn)
    #print("Server finished Loading model weights")
    f = open("server_info/velocity.bin",'rb')
    load_v = {}
    while True:
      try:
        t1 = pickle.load(f)
        t2 = pickle.load(f)
        if type(t1) == str:
            #print(t1)
            load_v[t1] = t2
        else:
            #print(t2)
            load_v[t2] = t1
      except EOFError:
        break
    f.close()
    #########################################
    ###     Server reads aggregated       ###
    ###     gradients.                    ###
    #########################################
    load_grads = optim.read_bin("gradients/grads_bin.bin")
    optim_config={'learning_rate': c_lr,
                      'momentum': 0.9,
                    }
    optim_configs={}
    for p in load_w:
        d = {k: v for k, v in optim_config.items()}
        d['velocity'] = load_v[p]
        optim_configs[p] = d
    #lr_decay=0.95

    # update
    f = open("server_info/velocity.bin",'wb')
    for p, w in load_w.items():
        dw = load_grads[p]
        config = optim_configs[p]
        next_w, next_config = optim.sgd_momentum(w, dw, config)
        load_w[p] = next_w
        optim_configs[p] = next_config
        pickle.dump(p, f)
        pickle.dump(next_config['velocity'], f)
    f.close()
    #print("Server updates the model weights")
    optim.write_bin('weights/weight_bin.bin', load_w)
    



