import numpy as np
import math

def l(i, n, t, T):
    """
    Return value of weight function

    @type  i: number
    @param i: index of node to which coefficient belongs
    @type  n: number
    @param n: number of nodes to approximate
    @type  t: float
    @param t: value of parameter
    @type  T: np.narray
    @param T: array of parameter values corresponding to each point
    """
    return np.prod((t - np.delete(T, i)))/np.prod(T[i] - np.delete(T, i))


def generate_T(U, normalized=True):
    """

    """
    (n, __) = U.shape

    if normalized == True:
        return np.arange(n)
    else:
        distances = np.array([math.sqrt(float((np.matrix(U[i+1])-np.matrix(U[i]))*(np.matrix(U[i+1])-np.matrix(U[i])).transpose())) for i in xrange(n-1)],
                             dtype=float)
        return np.insert(np.cumsum(distances), 0, 0)


def get_normalized_L(U, t):
    """

    """
    (n, ___) = U.shape

    T = generate_T(U, normalized=True)

    return [l(i, n, t, T) for i in xrange(n)]


def P(U, parameter_range):
    """

    """
    (n, ___) = U.shape
    return np.array([ np.sum(np.array(
        [get_normalized_L(U, t)[i]*U[i] for i in xrange(n)])) for t in parameter_range ])



number_of_points = 10
U = np.hstack((np.arange(number_of_points).reshape((number_of_points, 1)),
               np.random.rand(number_of_points, 1)*10))

print U
print P(U, np.arange(start=0.0, stop=float(number_of_points), step=float(0.1)))