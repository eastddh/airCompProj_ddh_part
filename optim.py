import numpy as np
import pandas as pd
"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:
def update(w, dw, config=None):
Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.
Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.
NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.
For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config['velocity'] = v

    return next_w, config



def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    eps, learning_rate = config['epsilon'], config['learning_rate']
    beta, v = config['decay_rate'], config['cache']
    # RMSprop
    v = beta * v + (1 - beta) * (dw * dw)        # exponential weighted average
    next_w = w - learning_rate * dw / (np.sqrt(v) + eps)
    config['cache'] = v
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
    eps, learning_rate = config['epsilon'], config['learning_rate']
    beta1, beta2 = config['beta1'], config['beta2']
    m, v, t = config['m'], config['v'], config['t']
    # Adam
    t = t + 1
    m = beta1 * m + (1 - beta1) * dw          # momentum
    mt = m / (1 - beta1**t)                   # bias correction
    v = beta2 * v + (1 - beta2) * (dw * dw)   # RMSprop
    vt = v / (1 - beta2**t)                   # bias correction
    next_w = w - learning_rate * mt / (np.sqrt(vt) + eps)
    # update values
    config['m'], config['v'], config['t'] = m, v, t
    return next_w, config

"""
load data
"""
class myDataset:
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.data = pd.read_csv(csv_file)
        n = len(self.data)
        self.old_pos = self.data.iloc[:,0:2]
        self.pos = self.data.iloc[:,0:2]
        self.old_pos = np.asarray(self.old_pos).astype('float').reshape(-1, 2)
        # rescale
        max_lat = max(self.old_pos[:,0])
        min_lat = min(self.old_pos[:,0])

        max_lon = max(self.old_pos[:,1])
        min_lon = min(self.old_pos[:,1])

        self.pos = self.data.iloc[:,0:2]
        self.pos = np.asarray(self.pos).astype('float').reshape(-1, 2)
        self.pos[:,0] = (self.pos[:,0] - min_lat)/(max_lat-min_lat)
        self.pos[:,1] = (self.pos[:,1] - min_lon)/(max_lon-min_lon)
        
        RSSI = self.data.iloc[0:n,2:3]
        self.RSSI = np.asarray(RSSI).astype('float').reshape(-1, 1)

    def __len__(self):
        return int(len(self.data))

    def __getitem__(self, idx):
        pos = np.array(self.pos[idx])
        RSSI = np.array(self.RSSI[idx])
        sample = {'positions':pos , 'RSSI': RSSI}
        return sample

##layers
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)
    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    #print('affine forward')
    out = None
    dim_size = x[0].shape
    X = x.reshape(x.shape[0], np.prod(dim_size))
    out = X.dot(w) + b
    cache = (x, w, b)
    #print(x.shape, 'x shape')
    #print(out.shape, 'out shape')
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dim_shape = np.prod(x[0].shape)
    N = x.shape[0]
    X = x.reshape(N, dim_shape)
    #print(dout.shape, 'dout shape')
    # input gradient
    #dx = dout.dot(w.T)
    #print('affine backward')
    #print(x.shape,'x shape')
    #print(w.shape,'w shape')
    #print(b.shape,'b shape')
    #print(dout.shape, 'dout shape')
    dx = np.dot(dout,(w.T))
    dx = dx.reshape(x.shape)
    # weight gradient
    #dw = X.T.dot(dout)
    dw = np.dot(X.T, dout)
    # bias gradient
    db = dout.sum(axis=0)
    return dx, dw, db


def Leaky_relu_forward(x):
    cache = x
    #print('relu forward')
    out = np.where(x > 0, x, x * 0.01)
    #print(x.shape, 'x shape')
    #print(out.shape, 'out shape')
    return out, x

def Leaky_relu_backward(dout, cache, alpha = 0.01):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    #print('relu back')
    
    dx = dout * (x > 0)
    #print(x.shape, 'x shape')
    #print(dx.shape, 'dx shape')
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    #print('batch forward')
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        mu = x.mean(axis=0)
        var = x.var(axis=0) + eps
        std = np.sqrt(var)
        z = (x - mu)/std
        out = gamma * z + beta

        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * (std**2)
        # save values for backward call
        cache={'x':x,'mean':mu,'std':std,'gamma':gamma,'z':z,'var':var,'axis':0}
    elif mode == 'test':
        out = gamma * (x - running_mean) / np.sqrt(running_var + eps) + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache



def batchnorm_backward(dout, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.
    Inputs / outputs: Same as batchnorm_backward
    """
    #print('batch backward')
    dx, dgamma, dbeta = None, None, None
    dbeta = dout.sum(axis=cache['axis'])
    dgamma = np.sum(dout * cache['z'], axis=cache['axis'])

    N = dout.shape[0]
    z = cache['z']
    dfdz = dout * cache['gamma']                                    #[NxD]
    dfdz_sum = np.sum(dfdz,axis=0)                                  #[1xD]
    dx = dfdz - dfdz_sum/N - np.sum(dfdz * z,axis=0) * z/N          #[NxD]
    dx /= cache['std']

    return dx, dgamma, dbeta

def mse_loss(scores, y):
    """
    compute loss and gradient of mse loss
    """
    loss = np.mean(np.square(y - scores),axis = 0)
    dout = 2*(scores - y)/len(y)
    #print(dout.shape,'mse dout shape')
    #print(y.shape,'mse y shape')
    #print(scores.shape,'mse scores shape')
    #print(loss.shape,'mse loss shape')
    return loss, dout

"""Helper functions"""

def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
    bn_cache = None
    # affine layer
    out, fc_cache = affine_forward(x,w,b)
    # batch/layer norm
    
    out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
       
    # relu
    out, relu_cache = Leaky_relu_forward(out)
   
    return out, (fc_cache, bn_cache, relu_cache)

def affine_norm_relu_backward(dout, cache ):
    fc_cache, bn_cache, relu_cache = cache

    # relu
    dout = Leaky_relu_backward(dout, relu_cache)
    # batch/layer norm
    dgamma, dbeta = None, None
    dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)   
    
    # affine layer
    dx, dw, db = affine_backward(dout, fc_cache)
    return dx, dw, db, dgamma, dbeta

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be
    {affine - [batch norm] - relu } x (L - 1) - affine 
    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=2,output_dim = 1, reg=0.0,
                 weight_scale=1e-2, dtype=np.float64, seed=None, load_weights = None, load_bn = None):
        """
        Initialize a new FullyConnectedNet.
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        layers_dims = np.hstack([input_dim, hidden_dims, output_dim])
        if load_weights == None:
            for i in range(self.num_layers):
                self.params['W'+str(i+1)] = weight_scale*np.random.randn(layers_dims[i],layers_dims[i+1])
                self.params['b'+str(i+1)] = np.zeros(layers_dims[i+1])
            for i in range(self.num_layers-1):
                self.params['gamma'+str(i+1)] = np.ones(layers_dims[i+1])
                self.params['beta' +str(i+1)] = np.zeros(layers_dims[i+1])
            self.bn_params = []
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        else:
            #print('Loading model weight')
            for i in range(self.num_layers):
                self.params['W'+str(i+1)] = load_weights['W'+str(i+1)]
                self.params['b'+str(i+1)] = load_weights['b'+str(i+1)]
            for i in range(self.num_layers-1):
                self.params['gamma'+str(i+1)] = load_weights['gamma'+str(i+1)]
                self.params['beta' +str(i+1)] = load_weights['beta' +str(i+1)]
            self.bn_params = load_bn
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
    
    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.

        for bn_param in self.bn_params:
            bn_param['mode'] = mode
        scores = None
        x = X
        caches = []
        gamma, beta, bn_params = None, None, None
        for i in range(self.num_layers-1):
            w = self.params['W'+str(i+1)]
            b = self.params['b'+str(i+1)]
            
            gamma = self.params['gamma'+str(i+1)]
            beta  = self.params['beta'+str(i+1)]
            bn_params = self.bn_params[i]
            x, cache = affine_norm_relu_forward(x,w,b, gamma, beta, bn_params)
            caches.append(cache)
        w = self.params['W'+str(self.num_layers)]
        b = self.params['b'+str(self.num_layers)]
        scores, cache = affine_forward(x,w,b)
        caches.append(cache)
        #print(mode, 'model mode')
        #print(scores[0:5],len(scores), 'scores in model loss')
        #print(y[0:5],len(y), 'y in model loss')
        #tL, tG = mse_loss(scores[0:5], y[0:5])
        #print('test mse',tL )
        #print('test grad',tG )
        # If test mode return early
        if mode == 'test':
            return scores

        # calculate loss
        loss, grads = 0.0, {}
        loss, mse_grad = mse_loss(scores, y)
        for i in range(self.num_layers):
            w = self.params['W'+str(i+1)]
            loss += 0.5 * self.reg * np.sum(w * w) 

        # calculate gradients
        dout = mse_grad
        dout, dw, db = affine_backward(dout, caches[self.num_layers - 1])
        grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db

        for i in range(self.num_layers - 2, -1, -1):
            dx, dw, db, dgamma, dbeta = affine_norm_relu_backward(dout, caches[i])
            grads['gamma'+str(i+1)] = dgamma
            grads['beta' +str(i+1)] = dbeta
            grads['W' + str(i + 1)] = dw + self.reg * self.params['W' + str(i + 1)]
            grads['b' + str(i + 1)] = db
            dout = dx

        return loss, grads

class Solver(object):
    """

    Example usage might look something like this:
    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()
    A Solver works on a model object that must conform to the following API:
    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.
    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:
      Inputs:
      - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
        label for X[i].
      Returns:
      If y is None, run a test-time forward pass and return:
      - scores: Array of shape (N, C) giving classification scores for X where
        scores[i, c] gives the score of class c for X[i].
      If y is not None, run a training time forward and backward pass and
      return a tuple of:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.
        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val': Array, shape (N_val,) of labels for validation images
        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every
          print_every iterations.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - num_train_samples: Number of training samples used to check training
          accuracy; default is 1000; set to None to use entire training set.
        - num_val_samples: Number of validation samples to use to check val
          accuracy; default is None, which uses the entire validation set.
        - checkpoint_name: If not None, then save model checkpoints here every
          epoch.
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)

        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()


    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_pred = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d


    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        #num_train = self.X_train.shape[0]
        #batch_mask = np.random.choice(num_train, self.batch_size)
        #X_batch = self.X_train[batch_mask]
        #y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(self.X_train, self.y_train)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config


    def _save_checkpoint(self):
        if self.checkpoint_name is None: return
        checkpoint = {
          'model': self.model,
          'update_rule': self.update_rule,
          'lr_decay': self.lr_decay,
          'optim_config': self.optim_config,
          'batch_size': self.batch_size,
          'num_train_samples': self.num_train_samples,
          'num_val_samples': self.num_val_samples,
          'epoch': self.epoch,
          'loss_history': self.loss_history,
          'train_acc_history': self.train_acc_history,
          'val_acc_history': self.val_acc_history,
          'val_predict': self.val_pred,
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)


    def check_accuracy(self, X, y, num_samples=None, batch_size=500):
        """
        Check accuracy of the model on the provided data.
        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.
        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            #print(i,'check accuracy')
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(scores)
        y_pred = np.hstack(y_pred)
        #test_mse,_ = mse_loss(y_pred[0:5],y[0:5])
        #print(test_mse)
        acc, _ = mse_loss(y_pred, y[start:end])

        return acc, y_pred


    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = 10
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) mse: %f' % (
                       t + 1, num_iterations, self.loss_history[-1]))

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc, _ = self.check_accuracy(self.X_train, self.y_train,
                    num_samples=self.num_train_samples, batch_size=len(self.y_train))
                val_acc, self.val_pred = self.check_accuracy(self.X_val, self.y_val,
                    num_samples=self.num_val_samples, batch_size=len(self.y_val))
                print(train_acc,'train mse')
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose and  self.epoch % self.print_every == 0:
                    print('(Epoch %d / %d) train mse: %f; val_mse: %f' % (
                           self.epoch, self.num_epochs, train_acc, val_acc))

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        #self.model.params = self.best_params
        val_acc, self.val_pred = self.check_accuracy(self.X_val, self.y_val,
                    num_samples=self.num_val_samples, batch_size=len(self.y_val))
class myDataset:
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.data = pd.read_csv(csv_file)
        n = len(self.data)
        self.old_pos = self.data.iloc[:,0:2]
        self.pos = self.data.iloc[:,0:2]
        self.old_pos = np.asarray(self.old_pos).astype('float').reshape(-1, 2)
        # rescale
        max_lat = max(self.old_pos[:,0])
        min_lat = min(self.old_pos[:,0])

        max_lon = max(self.old_pos[:,1])
        min_lon = min(self.old_pos[:,1])

        self.pos = self.data.iloc[:,0:2]
        self.pos = np.asarray(self.pos).astype('float').reshape(-1, 2)
        self.pos[:,0] = (self.pos[:,0] - min_lat)/(max_lat-min_lat)
        self.pos[:,1] = (self.pos[:,1] - min_lon)/(max_lon-min_lon)
        
        RSSI = self.data.iloc[0:n,2:3]
        self.RSSI = np.asarray(RSSI).astype('float').reshape(-1, 1)

    def __len__(self):
        return int(len(self.data))

    def __getitem__(self, idx):
        pos = np.array(self.pos[idx])
        RSSI = np.array(self.RSSI[idx])
        sample = {'positions':pos , 'RSSI': RSSI}
        return sample
