import numpy as np

def norm(   
        A: np.ndarray,
        type = 'fro') -> float:
    if A.ndim == 2:
        if type=='fro':
            ret = np.sqrt(np.sum(np.power(A, 2)))
            # ret = np.sqrt(np.trace(A @ A.T))
        elif type == 2:
            # By the min-max theorem, the 2-norm matrix is the largest singular value.
            ret =  np.linalg.svd(A)[1][0]
        elif type == 1:
            ret = np.max(np.sum(A, axis=0))
        elif type == np.inf:
            ret = np.max(np.sum(A, axis=1))
        else:
            raise ValueError("Invalid norm type for matrices.")
    else:
        raise ValueError("Invalid matrix dimension (must be 2D).")
    
    return ret

def cosineDistance(a, b):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # It is only defined for vectors.
    if a.ndim == 1 and b.ndim == 1:
        Sc = np.dot(a, b) / (np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2))
    return 1 - Sc