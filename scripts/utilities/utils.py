import numpy as np
import functools


def to_numpy(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        ret = fun(*args, **kwargs)
        if np.isscalar(ret):
            ret = np.array([ret])
        if not isinstance(ret, np.ndarray):
            ret = np.array(ret)
        return ret
    return wrapper


def check_butcher_sum(A, b, c, s):
    tol = 1e-5
    valid_b = np.sum(b) == 1
    valid_c = np.sum(
        np.abs(np.array([np.sum(row) for row in A]) - c)) < tol
    valid_size = len(b) == s and A.shape[0] == s and A.shape[1] == s

    assert valid_b, "invalid b component of butcher array"
    assert valid_c, "invalid c component of butcher array"
    assert valid_size, "array sizes not compatible"


def check_explicit_array(A, semi=False):
    if semi:
        return np.array_equal(A, np.tril(A))
    else:
        return np.array_equal(A, np.tril(A, -1))


def integral(g, j, p):
    deg = (p+1) // 2
    nodes, weights = np.polynomial.legendre.leggauss(deg=deg)
    nodes = 0.5*(nodes+1)
    weights = 0.5 * weights
    return np.sum(np.array([weights[i] * g(nodes[i], j) for i in range(deg)]))


def build_sequences(data, window, stride=1, telescope=1):
    #data should have shape (latent_dim, timesteps)
    
    assert window % stride == 0
    
    dataset = []
    target = []
        
    for idx in np.arange(0,data.shape[1]-window-telescope,stride):
        dataset.append(np.transpose(data[:,idx:idx+window]))
        target.append(np.transpose(data[:,idx+window:idx+window+telescope]))
    
    return np.array(dataset),np.array(target)


# X_train = np.array([])
# Y_train = np.array([])

# X = np.array([[0,1,2,3,4,5,6],
#                [10,11,12,13,14,15,16],
#                [20,21,22,23,24,25,26]])

# X_temp, Y_temp = build_sequences(X,4)


# X_train = X_temp
# Y_train = Y_temp

# X = np.array([[-0,-1,-2,-3,-4,-5,-6],
#                [-10,-11,-12,-13,-14,-15,-16],
#                [-20,-21,-22,-23,-24,-25,-26]])

# X_temp, Y_temp = build_sequences(X,4)


# X_train = np.concatenate((X_train,X_temp),0)
# Y_train = np.concatenate((Y_train,Y_temp),0)

# dim_x: (n_sample, latent_dim, timesteps)



#print(X_train.shape)
# print(Y_train)
    
    