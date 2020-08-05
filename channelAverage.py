import pickle
import sys
import numpy as np

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


clients = []
for num in range(numsClient):
    f=open("gradients\grads_bin"+str(num+1)+".bin","rb")
    clients.append(f)
    aaverage_G['W3'] += pickle.load(f)
    aaverage_G['b3'] += pickle.load(f)
    aaverage_G['gamma2'] += pickle.load(f)
    aaverage_G['beta2'] += pickle.load(f)
    aaverage_G['W2'] += pickle.load(f)
    aaverage_G['b2'] += pickle.load(f)
    aaverage_G['gamma1'] += pickle.load(f)
    aaverage_G['beta1'] += pickle.load(f)
    aaverage_G['W1'] += pickle.load(f)
    aaverage_G['b1'] += pickle.load(f)
    f.close()
    

for grad in aaverage_G:
    aaverage_G[grad] /= float(numsClient)

f = open('gradients\grads_bin.bin','wb')
for grad in aaverage_G:
  #print(grad)
  pickle.dump(aaverage_G[grad], f)
f.close()
print('Simulate the channel aggregation')




