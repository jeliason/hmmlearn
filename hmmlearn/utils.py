import numpy as np
from scipy.misc import logsumexp


def normalize(a, axis=None):
    """Normalizes the input array so that it sums to 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum


def log_normalize(a, axis=None):
    """Normalizes the input array so that the exponent of the sum is 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_lse = logsumexp(a, axis)
    a -= a_lse[:, np.newaxis]


def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {0:d} samples in lengths array {1!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]


def log_mask_zero(a):
    """Computes the log of input probabilities masking divide by zero in log.

    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.

    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        a_log = np.log(a)
        a_log[a <= 0] = 0.0
        return a_log

def log_multivariate_poisson_density(X, means) :
  # # modeled on log_multivariate_normal_density from sklearn.mixture
  n_samples, n_dim = X.shape
  # -lambda + k log(lambda) - log(k!)
  log_means = np.where(means > np.log(1e-3), np.log(means), np.log(1e-3))
  lpr =  np.dot(X, log_means.T)
  lpr = lpr - np.sum(means,axis=1) # rates for all elements are summed and then broadcast across the observation dimenension
  log_factorial = np.sum(gammaln(X + 1), axis=1)
  lpr = lpr - log_factorial[:,None] # logfactobs vector broad cast across the state dimension
  return lpr