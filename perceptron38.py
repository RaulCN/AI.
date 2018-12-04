#Errinho bobo corrigido, agora é entender oq ele faz
#https://gist.githubusercontent.com/alexland/0cb8a7e81705e6cb6c14/raw/d12e0592936d6bbb5413de7982b416c1b747e9c7/softmax-mlp.py

'''
estas 2 funções assumem este tipo de arquitetura MLP:
  (i) um problema de classificação modelado com função de ativação softmax
  (ii) codifique a camada de saída w/ 1-of-N 
    > Então, para dados brutos como este:
        .3, .2, .6, .1, 'class I'
        .6, .1, .8, .4, 'class II'
        .5, .2, .7, .3, 'class III'
        
      recode it for intput to a softmax MLP like so:
        .3, .2, .6, .1, 1, 0, 0
        .6, .1, .8, .4, 0, 1, 0
        .5, .2, .7, .3, 0, 0, 1
        
    O softmax requer
      > MLP network to have 3 neurons in the output layer, and
      > the sum of the output layer equals 1.0;
      
    > exception for 2-class problems, use 1-of-(N-1) encoding
  (iii) testing error measured w/ cross-entropy error (see 
  http://jamesmccaffrey.wordpress.com/2014/04/25/neural-network-cross-entropy-error/)

'''

import numpy as NP
from cmath import log10


def sm(v):
  '''
  returns: vector (or scalar if v is scalar) of same len
    as v, representing probabilities for respective class,
    ie, a categorical probability distribution
  pass in: v, MLP output layer as numpy array;
  this the softmax fn, which transforms your output-layer
    vector to a vector (of probabilities) whose values sum to 1
    
  >>> # output from MLP:
  >>> v = NP.array([2.0 -3.0, 0.0])
  >>> res = sm(v)
  >>> target = 1.0
  >>> from numpy.testing import assert_almost_equal
  >>> assert_almost_equal(res.sum(), target)  
  
  '''
  n = NP.exp(v)
  d = n.sum()
  return n/d


def cee(tv, pv):
  '''
  returns:
  pass in:
    (i)  tv, numpy 1D array, output 
    (ii) pv, numpy 1D array returned from sm above;
  this fn calculates cross-entropy error which,
    in this context, is the better error metric vs SSD
    
  >>> tv = NP.array([1., 0, 0])
  >>> pv = NP.array([ 0.876,  0.006,  0.118])
  >>> res = cee(tv, pv)
  >>> target = -0.057693951955
  >>> assert_almost_equal(res, target)
  
  '''
  cce_in = zip(tv, pv)
  fnx = lambda a, b: a * log10(b)
  return NP.sum(fnx(*itm) for itm in cce_in).real
  
  
  
if __name__ == '__main__':
  '''
  rexecutando este script a partir da linha de comando w/ the '-v' flag 
 irá executar os doctests e mostrar a saída no terminal
  '''
  import doctest
  doctest.testmod()
