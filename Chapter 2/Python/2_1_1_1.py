import numpy as np
import math


def l(i, t, T):
    """
    Return value of weight function

    @type  i: number
    @param i: index of node to which coefficient belongs
    @type  t: float
    @param t: value of parameter
    @type  T: np.narray
    @param T: array of parameter values corresponding to each point
    """
    return np.prod((t - np.delete(T, i)))/np.prod(T[i] - np.delete(T, i))


def generate_T(U, normalized=True):
    """
    Returns vector of parameter values that correspond to each 
    node in U
    
    @type  U: np.narray
    @param U: vector of nodes
    @type  normalized: bool
    @param normalized: True = [0,1.0,2.0...] else [0,d_1,d_1 + d_2,...] where d
    is distance between points
    """
    (n, __) = U.shape

    if normalized == True:
        return np.arange(n)
    else:
        distances = np.array([math.sqrt(
            float((np.matrix(U[i+1])-np.matrix(U[i]))*
                  (np.matrix(U[i+1])-np.matrix(U[i])).transpose()))
                              for i in xrange(n-1)],
                             dtype=float)
        return np.insert(np.cumsum(distances), 0, 0)


def get_normalized_L(U, t):
    """
    Returns vector of node's weights for specific value of
    parameter t
    
    @type  U: np.narray
    @param U: vector of nodes
    @type  t: float
    @param t: parameter to calculate weight's vector for
    """
    (n, __) = U.shape

    T = generate_T(U, normalized=True)

    return [l(i, t, T) for i in xrange(n)]


def P(U, parameter_range):
    """
    Returns array of points that should correspond to
    parameter_range provided. It could be considered as
    result of interpolation.
    
    @type  U: np.narray
    @param U: vector of nodes
    @type  parameter_range: np.array
    @param parameter_range: array of parameters to map interpolation function to
    """
    (n, vector_length) = U.shape
    
    array_to_return = np.zeros(shape=(parameter_range.shape[0],vector_length))
    
    for parameter_index in xrange(parameter_range.shape[0]):
        t = parameter_range[parameter_index]
        normalized_L = get_normalized_L(U,t)
        for i in xrange(n):
            array_to_return[parameter_index] += normalized_L[i]*U[i]

    return array_to_return


number_of_points = 10

nodes_to_approximate = np.hstack(
    (np.arange(number_of_points).reshape((number_of_points, 1)),
     np.random.rand(number_of_points, 1)*10))

p = P(nodes_to_approximate,
      np.arange(start=0.0, stop=float(number_of_points-1+0.1), step=float(0.1)))

from pylab import plot, show
plot(nodes_to_approximate[..., 0],
     nodes_to_approximate[..., 1],
     'o',
     p[..., 0],
     p[..., 1])

show()
