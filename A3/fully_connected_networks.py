"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
from a3_helper import svm_loss, softmax_loss
from eecs598 import Solver

def hello_fully_connected_networks():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from fully_connected_networks.py!')


class Linear(object):

  @staticmethod
  def forward(x, w, b):
    """
    Computes the forward pass for an linear (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
    - w: A tensor of weights, of shape (D, M)
    - b: A tensor of biases, of shape (M,)
    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the linear forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    # Replace "pass" statement with your code
    t = x.view(x.shape[0], -1).cuda()
    out = t.mm(w.cuda())+b.cuda()
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    cache = (x, w, b)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Computes the backward pass for an linear layer.
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
    #############################################################################
    # TODO: Implement the linear backward pass.                                 #
    #############################################################################
    # # Replace "pass" statement with your code
    # dout = dout.cuda()
    # w = w.cuda()
    dx = dout.mm(w.t())
    dx = dx.view(x.size())
    t = x.view(x.shape[0], -1)
    dw = torch.mm(t.t(), dout)
    db = dout.sum(axis = 0)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return dx, dw, db


class ReLU(object):

  @staticmethod
  def forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Input; a tensor of any shape
    Returns a tuple of:
    - out: Output, a tensor of the same shape as x
    - cache: x
    """
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    # You should not change the input tensor with an in-place operation.        #
    #############################################################################
    # Replace "pass" statement with your code
    out = x.clamp(min = 0)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    cache = x
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    # You should not change the input tensor with an in-place operation.        #
    #############################################################################
    # Replace "pass" statement with your code
    mask = torch.zeros_like(x)
    mask[x>0] = 1
    dx = mask * dout
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return dx


class Linear_ReLU(object):

  @staticmethod
  def forward(x, w, b):
    """
    Convenience layer that performs an linear transform followed by a ReLU.

    Inputs:
    - x: Input to the linear layer
    - w, b: Weights for the linear layer
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = Linear.forward(x, w, b)
    out, relu_cache = ReLU.forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the linear-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = ReLU.backward(dout, relu_cache)
    dx, dw, db = Linear.backward(da, fc_cache)
    return dx, dw, db


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  The architecure should be linear - relu - linear - softmax.
  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to PyTorch tensors.
  """

  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
         weight_scale=1e-3, reg=0.0, dtype=torch.double, device='cpu'):
    """
    Initialize a new network.
    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'
    """
    self.params = {}
    self.reg = reg

    ###########################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights   #
    # should be initialized from a Gaussian centered at 0.0 with              #
    # standard deviation equal to weight_scale, and biases should be          #
    # initialized to zero. All weights and biases should be stored in the     #
    # dictionary self.params, with first layer weights                        #
    # and biases using the keys 'W1' and 'b1' and second layer                #
    # weights and biases using the keys 'W2' and 'b2'.                        #
    ###########################################################################
    # Replace "pass" statement with your code
    self.params['W1'] = torch.randn(input_dim, hidden_dim).double() * weight_scale
    self.params['b1']= torch.zeros(hidden_dim).double() 
    self.params['W2'] = torch.randn(hidden_dim, num_classes).double() * weight_scale
    self.params['b2'] = torch.zeros(num_classes).double()
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################


  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'params': self.params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))

  def load(self, path, dtype, device):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.reg = checkpoint['reg']
    for p in self.params:
      self.params[p] = self.params[p].type(dtype).to(device)
    print("load checkpoint file: {}".format(path))


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Tensor of input data of shape (N, d_1, ..., d_k)
    - y: int64 Tensor of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Tensor of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.
    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    scores = None
    ###########################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the   #
    # class scores for X and storing them in the scores variable.             #
    ###########################################################################
    # Replace "pass" statement with your code
    X = X.view(X.shape[0], -1)
    # first_layer_score = X.mm(self.params['W1'])+self.params['b1']
    # first_layer_score_relu = first_layer_score.clamp(min=0)
    # scores = first_layer_score_relu.mm(self.params['W2'])+self.params['b2']
    linear1, linear1_cache = Linear_ReLU.forward(x = X, w = self.params['W1'], b = self.params['b1'])
    scores, linear2_cache = Linear.forward(x = linear1, w = self.params['W2'], b = self.params['b2']) 
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, grads = 0, {}
    ###########################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss #
    # in the loss variable and gradients in the grads dictionary. Compute data#
    # loss using softmax, and make sure that grads[k] holds the gradients for #
    # self.params[k]. Don't forget to add L2 regularization!                  #
    #                                                                         #
    # NOTE: To ensure that your implementation matches ours and you pass the  #
    # automated tests, make sure that your L2 regularization does not include #
    # a factor of 0.5.                                                        #
    ###########################################################################
    # Replace "pass" statement with your code
    loss, dout = softmax_loss(scores, y)
    loss += self.reg * torch.sum(self.params['W1']**2)
    loss += self.reg * torch.sum(self.params['W2']**2)

    dx2, grads['W2'], grads['b2'] = Linear.backward(dout, linear2_cache)
    _, grads['W1'], grads['b1'] = Linear_ReLU.backward(dx2, linear1_cache)

    grads['W2'] += 2 * self.reg * self.params['W2']
    grads['W1'] += 2 * self.reg * self.params['W1']

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

    return loss, grads

# class FullyConnectedNet(object):
#   """
#   A fully-connected neural network with an arbitrary number of hidden layers,
#   ReLU nonlinearities, and a softmax loss function.
#   For a network with L layers, the architecture will be:

#   {linear - relu - [dropout]} x (L - 1) - linear - softmax

#   where dropout is optional, and the {...} block is repeated L - 1 times.

#   Similar to the TwoLayerNet above, learnable parameters are stored in the
#   self.params dictionary and will be learned using the Solver class.
#   """

#   def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
#                dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
#                dtype=torch.float, device='cpu'):
#     """
#     Initialize a new FullyConnectedNet.

#     Inputs:
#     - hidden_dims: A list of integers giving the size of each hidden layer.
#     - input_dim: An integer giving the size of the input.
#     - num_classes: An integer giving the number of classes to classify.
#     - dropout: Scalar between 0 and 1 giving the drop probability for networks
#       with dropout. If dropout=0 then the network should not use dropout.
#     - reg: Scalar giving L2 regularization strength.
#     - weight_scale: Scalar giving the standard deviation for random
#       initialization of the weights.
#     - seed: If not None, then pass this random seed to the dropout layers. This
#       will make the dropout layers deteriminstic so we can gradient check the
#       model.
#     - dtype: A torch data type object; all computations will be performed using
#       this datatype. float is faster but less accurate, so you should use
#       double for numeric gradient checking.
#     - device: device to use for computation. 'cpu' or 'cuda'
#     """
#     self.use_dropout = dropout != 0
#     self.reg = reg
#     self.num_layers = 1 + len(hidden_dims)
#     self.dtype = dtype
#     self.params = {}

#     ############################################################################
#     # TODO: Initialize the parameters of the network, storing all values in    #
#     # the self.params dictionary. Store weights and biases for the first layer #
#     # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
#     # initialized from a normal distribution centered at 0 with standard       #
#     # deviation equal to weight_scale. Biases should be initialized to zero.   #
#     ############################################################################
#     # Replace "pass" statement with your code
    
#     # first layer
#     self.params['W1'] =  weight_scale * torch.randn(input_dim, hidden_dims[0], dtype=dtype, device=device)
#     self.params['b1'] = torch.zeros(hidden_dims[0], dtype=dtype, device=device)
    
#     # hidden layer
#     for i in range(0, len(hidden_dims)-1):
#       self.params["W"+ str(i+2)] = weight_scale * torch.randn(hidden_dims[i], hidden_dims[i+1], dtype=dtype, device=device)
#       self.params["b"+ str(i+2)] = torch.zeros(hidden_dims[i+1], dtype=dtype, device=device)
      
#     # final layer
#     self.params["W"+ str(self.num_layers)] = weight_scale * torch.randn(hidden_dims[-1], num_classes, dtype=dtype, device=device)
#     self.params["b"+ str(self.num_layers)] = torch.zeros(num_classes, dtype=dtype, device=device)
      
#     ############################################################################
#     #                             END OF YOUR CODE                             #
#     ############################################################################

#     # When using dropout we need to pass a dropout_param dictionary to each
#     # dropout layer so that the layer knows the dropout probability and the mode
#     # (train / test). You can pass the same dropout_param to each dropout layer.
#     self.dropout_param = {}
#     if self.use_dropout:
#       self.dropout_param = {'mode': 'train', 'p': dropout}
#       if seed is not None:
#         self.dropout_param['seed'] = seed


#   def loss(self, X, y=None):
#     """
#     Compute loss and gradient for the fully-connected net.
#     Input / output: Same as TwoLayerNet above.
#     """
#     X = X.to(self.dtype)
#     mode = 'test' if y is None else 'train'

#     # Set train/test mode for batchnorm params and dropout param since they
#     # behave differently during training and testing.
#     if self.use_dropout:
#       self.dropout_param['mode'] = mode
#     scores = None
#     ############################################################################
#     # TODO: Implement the forward pass for the fully-connected net, computing  #
#     # the class scores for X and storing them in the scores variable.          #
#     #                                                                          #
#     # When using dropout, you'll need to pass self.dropout_param to each       #
#     # dropout forward pass.                                                    #
#     ############################################################################
#     # Replace "pass" statement with your code
#     h_out = X
#     caches = [] # a list of caches that can be popped in back-prop
#     drop_cache = []
#     # for the first num_layer - 1, it is Linear-Relu layer
#     for i in range(0, self.num_layers - 1):
#       w = self.params['W' + str(i + 1)]
#       b = self.params['b' + str(i + 1)]
      
#       h_out, cache = Linear_ReLU.forward(h_out, w, b)
#       caches.append(cache)
#       # if self.use_dropout:
#       #   h_out, dropped = Dropout.forward(h_out, self.dropout_param)
#       #   drop_cache.append(dropped)
    
#     # then is the last layer
#     w = self.params['W' + str(self.num_layers)]
#     b = self.params['b' + str(self.num_layers)]
#     h_out, cache = Linear.forward(h_out, w, b)
#     caches.append(cache)
#     scores = h_out
#     ############################################################################
#     #                             END OF YOUR CODE                             #
#     ############################################################################

#     # If test mode return early
#     if mode == 'test':
#       return scores

#     loss, grads = 0.0, {}
#     ############################################################################
#     # TODO: Implement the backward pass for the fully-connected net. Store the #
#     # loss in the loss variable and gradients in the grads dictionary. Compute #
#     # data loss using softmax, and make sure that grads[k] holds the gradients #
#     # for self.params[k]. Don't forget to add L2 regularization!               #
#     # NOTE: To ensure that your implementation matches ours and you pass the   #
#     # automated tests, make sure that your L2 regularization includes a factor #
#     # of 0.5 to simplify the expression for the gradient.                      #
#     ############################################################################
#     # Replace "pass" statement with your code
#     loss, dout = softmax_loss(h_out, y)
#     # also remember the regularization term
#     for i in range(self.num_layers):
#       w = self.params['W' + str(i+1)]
#       loss += self.reg * torch.sum(w ** 2)
    
#     # for the last layer, it is only Linear forward
#     dout, dw, db = Linear.backward(dout, caches.pop())
#     # w = self.params['W' + str(num_layers)]
#     # b = self.params['b' + str(num_layers)]
#     grads['W' + str(self.num_layers)] = dw + 2 * self.reg * self.params['W' + str(self.num_layers)]
#     grads['b' + str(self.num_layers)] = db
    
#     for i in range(0, self.num_layers-1):
#       # go to index 0
#       if self.use_dropout:
#         dout = Dropout.backward(dout, drop_cache.pop())
#       dout, dw, db = Linear_ReLU.backward(dout, caches.pop())
#       grads['W' + str(self.num_layers -1 - i)] = dw + 2 * self.reg * self.params['W' + str(self.num_layers -1 - i)]
#       grads['b' + str(self.num_layers -1 - i)] = db
    
#     ############################################################################
#     #                             END OF YOUR CODE                             #
#     ############################################################################

#     return loss, grads    

class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function.
  For a network with L layers, the architecture will be:

  {linear - relu - [dropout]} x (L - 1) - linear - softmax

  where dropout is optional, and the {...} block is repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
               dtype=torch.float, device='cpu'):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving the drop probability for networks
      with dropout. If dropout=0 then the network should not use dropout.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'
    """
    self.use_dropout = dropout != 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution centered at 0 with standard       #
    # deviation equal to weight_scale. Biases should be initialized to zero.   #
    ############################################################################
    # Replace "pass" statement with your code
    for i in range(self.num_layers):
      name_W = 'W'+'{}'.format(i+1)
      name_b = 'b'+'{}'.format(i+1)
      if i==0:
        self.params[name_W] = torch.randn(input_dim, hidden_dims[i], dtype=dtype, device=device) * weight_scale
        self.params[name_b] = torch.zeros(hidden_dims[i], dtype=dtype, device=device)

      elif i == self.num_layers-1:
        self.params[name_W] = torch.randn(hidden_dims[-1], num_classes, dtype=dtype, device=device) * weight_scale
        self.params[name_b] = torch.zeros(num_classes, dtype=dtype, device=device)
      
      else:
        self.params[name_W] = torch.randn(hidden_dims[i-1], hidden_dims[i], dtype=dtype, device=device) * weight_scale
        self.params[name_b] = torch.zeros(hidden_dims[i])
    # first layer
    # self.params['W1'] =  weight_scale * torch.randn(input_dim, hidden_dims[0], dtype=dtype, device=device)
    # self.params['b1'] = torch.zeros(hidden_dims[0], dtype=dtype, device=device)
    
    # # hidden layer
    # for i in range(0, len(hidden_dims)-1):      
    #   name_W = 'W'+'{}'.format(i+2)
    #   print(name_W)
    #   print("W"+ str(i+2))
    #   self.params["W"+ str(i+2)] = weight_scale * torch.randn(hidden_dims[i], hidden_dims[i+1], dtype=dtype, device=device)
    #   self.params["b"+ str(i+2)] = torch.zeros(hidden_dims[i+1], dtype=dtype, device=device)
      
    # # final layer
    # self.params["W"+ str(self.num_layers)] = weight_scale * torch.randn(hidden_dims[-1], num_classes, dtype=dtype, device=device)
    # self.params["b"+ str(self.num_layers)] = torch.zeros(num_classes, dtype=dtype, device=device)
    # print(self.params.keys())
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed


  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
      'num_layers': self.num_layers,
      'use_dropout': self.use_dropout,
      'dropout_param': self.dropout_param,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))


  def load(self, path, dtype, device):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = dtype
    self.reg = checkpoint['reg']
    self.num_layers = checkpoint['num_layers']
    self.use_dropout = checkpoint['use_dropout']
    self.dropout_param = checkpoint['dropout_param']

    for p in self.params:
      self.params[p] = self.params[p].type(dtype).to(device)

    print("load checkpoint file: {}".format(path))

  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.
    Input / output: Same as TwoLayerNet above.
    """
    X = X.to(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.use_dropout:
      self.dropout_param['mode'] = mode
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    ############################################################################
    # Replace "pass" statement with your code
    out = X
    forward_cache = []
    dropout_cache = []
    for i in range(self.num_layers-1):
      name_W = 'W'+'{}'.format(i+1)
      name_b = 'b'+'{}'.format(i+1)
      out, cache = Linear_ReLU.forward(x = out, w = self.params[name_W], b = self.params[name_b])
      forward_cache.append(cache)

      if self.use_dropout:
        out, cache = Dropout.forward(out, self.dropout_param)
        dropout_cache.append(cache)

    name_W = 'W'+'{}'.format(self.num_layers)
    name_b = 'b'+'{}'.format(self.num_layers)
    out, cache = Linear.forward(out, self.params[name_W], self.params[name_b])
    forward_cache.append(cache)

    scores = out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # Replace "pass" statement with your code
    loss, dout = softmax_loss(scores, y)
    for i in range(self.num_layers):
      name_W = 'W'+'{}'.format(i+1)
      loss +=  self.reg * torch.sum(self.params[name_W]**2)

    for i in range(self.num_layers, 0, -1):
      name_W = 'W'+'{}'.format(i)
      name_b = 'b' + '{}'.format(i)
      if i == self.num_layers:
        grads_x, grads[name_W], grads[name_b] = Linear.backward(dout, forward_cache.pop())
        grads[name_W] += 2 * self.reg * self.params[name_W]
      else:
        if self.use_dropout:
          grads_x = Dropout.backward(grads_x, dropout_cache.pop())
        grads_x, grads[name_W], grads[name_b] = Linear_ReLU.backward(grads_x, forward_cache.pop())
        grads[name_W] += 2 * self.reg * self.params[name_W]


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


def create_solver_instance(data_dict, dtype, device):
  model = TwoLayerNet(input_dim=3*32*32, hidden_dim=500)
  ##############################################################################
  # TODO: Use a Solver instance to train a TwoLayerNet that achieves at least  #
  # 50% accuracy on the validation set.                                        #
  ##############################################################################
  solver = None
  # Replace "pass" statement with your code
  data = data_dict
  solver = Solver(model, data, num_epochs=30, batch_size=100, print_every=100)
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return solver


def get_three_layer_network_params():
  ############################################################################
  # TODO: Change weight_scale and learning_rate so your model achieves 100%  #
  # training accuracy within 20 epochs.                                      #
  ############################################################################
  weight_scale = 7e-2   # Experiment with this!
  learning_rate = 5e-1  # Experiment with this!
  # Replace "pass" statement with your code
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################
  return weight_scale, learning_rate


def get_five_layer_network_params():
  ############################################################################
  # TODO: Change weight_scale and learning_rate so your model achieves 100%  #
  # training accuracy within 20 epochs.                                      #
  ############################################################################
  weight_scale = 1e-1  # Experiment with this!
  learning_rate = 5e-1 # Experiment with this!
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################
  return weight_scale, learning_rate


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    w = w.cuda()
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
  v = config.get('velocity', torch.zeros_like(w))

  next_w = None
  #############################################################################
  # TODO: Implement the momentum update formula. Store the updated value in   #
  # the next_w variable. You should also use and update the velocity v.       #
  #############################################################################
  # Replace "pass" statement with your code
  v = v.cuda()
  w = w.cuda()
  v = config['momentum']*v - config['learning_rate']*dw
  next_w = w + v
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
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
  config.setdefault('cache', torch.zeros_like(w))

  next_w = None
  ###########################################################################
  # TODO: Implement the RMSprop update formula, storing the next value of w #
  # in the next_w variable. Don't forget to update cache value stored in    #
  # config['cache'].                                                        #
  ###########################################################################
  # Replace "pass" statement with your code
  config['cache'] = config['decay_rate'] * config['cache'].cuda() + (1 - config['decay_rate'])*dw.cuda()**2
  next_w = w.cuda() - config['learning_rate']*dw/(torch.sqrt(config['cache'])+config['epsilon'])
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
  config.setdefault('m', torch.zeros_like(w))
  config.setdefault('v', torch.zeros_like(w))
  config.setdefault('t', 0)

  next_w = None
  #############################################################################
  # TODO: Implement the Adam update formula, storing the next value of w in   #
  # the next_w variable. Don't forget to update the m, v, and t variables     #
  # stored in config.                                                         #
  #                                                                           #
  # NOTE: In order to match the reference output, please modify t _before_    #
  # using it in any calculations.                                             #
  #############################################################################
  # Replace "pass" statement with your code
  config['t'] +=1
  config['m'] = config['beta1']*config['m'].cuda() + (1-config['beta1']) * dw.cuda()
  monmetnum1_unbias = config['m'] / (1-config['beta1'] ** config['t'])
  config['v'] = config['beta2']*config['v'].cuda() + (1-config['beta2'])*dw*dw
  momentum2_unbias = config['v']/(1-config['beta2']**config['t'])
  next_w = w.cuda() - config['learning_rate']*monmetnum1_unbias/(momentum2_unbias**0.5+config['epsilon'])
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  return next_w, config

class Dropout(object):

  @staticmethod
  def forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.
    Inputs:
    - x: Input data: tensor of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We *drop* each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not
      in real networks.
    Outputs:
    - out: Tensor of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.
    NOTE 2: Keep in mind that p is the probability of **dropping** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of keeping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
      torch.manual_seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
      ###########################################################################
      # TODO: Implement training phase forward pass for inverted dropout.       #
      # Store the dropout mask in the mask variable.                            #
      ###########################################################################
      # Replace "pass" statement with your code
      mask = (torch.randn_like(x)<p)/p
      out = x * mask
      ###########################################################################
      #                             END OF YOUR CODE                            #
      ###########################################################################
    elif mode == 'test':
      ###########################################################################
      # TODO: Implement the test phase forward pass for inverted dropout.       #
      ###########################################################################
      # Replace "pass" statement with your code
      out = x
      ###########################################################################
      #                             END OF YOUR CODE                            #
      ###########################################################################

    cache = (dropout_param, mask)

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.
    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from Dropout.forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
      ###########################################################################
      # TODO: Implement training phase backward pass for inverted dropout       #
      ###########################################################################
      # Replace "pass" statement with your code
      dx = mask * dout
      ###########################################################################
      #                            END OF YOUR CODE                             #
      ###########################################################################
    elif mode == 'test':
      dx = dout
    return dx
