import pickle
import sys
import numpy as np
import optim

if len(sys.argv) != 2:
    print("Error, require one number input. e.g. python3 channelAverage.py 2")
    sys.exit()
numsClient = int(sys.argv[1])

aaverage_G = {}
aaverage_G['W3'] = np.zeros([100,1])
aaverage_G['b3'] = np.zeros(1)
aaverage_G['gamma2'] = np.zeros(100)
aaverage_G['beta2'] = np.zeros(100)
aaverage_G['W2'] = np.zeros([100,100])
aaverage_G['b2'] = np.zeros(100)
aaverage_G['gamma1'] = np.zeros(100)
aaverage_G['beta1'] = np.zeros(100)
aaverage_G['W1'] = np.zeros([2,100])
aaverage_G['b1'] = np.zeros(100)


for num in range(numsClient):
    fn = "gradients/grads_bin"+str(num+1)+".bin"
    load_grad = optim.read_bin(fn)
    for para in load_grad:
        aaverage_G[para] += load_grad[para]
    

for grad in aaverage_G:
    aaverage_G[grad] /= float(numsClient)

optim.write_bin('gradients/grads_bin.bin', aaverage_G)




