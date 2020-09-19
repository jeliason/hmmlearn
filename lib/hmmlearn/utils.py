import numpy as np
<<<<<<< HEAD:hmmlearn/utils.py
from scipy.misc import logsumexp
from scipy.special import gammaln
=======
from scipy.special import logsumexp
>>>>>>> add35ff41052fc23e76fbd484dbed8ddcb1a452d:lib/hmmlearn/utils.py


def normalize(a, axis=None):
    """
    Normalizes the input array so that it sums to 1.

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
    """
    Normalizes the input array so that ``sum(exp(a)) == 1``.

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
    if axis is not None and a.shape[axis] == 1:
        # Handle single-state GMMHMM in the degenerate case normalizing a
        # single -inf to zero.
        a[:] = 0
    else:
        with np.errstate(under="ignore"):
            a_lse = logsumexp(a, axis, keepdims=True)
        a -= a_lse


def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {:d} samples in lengths array {!s}"
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
<<<<<<< HEAD:hmmlearn/utils.py
        a_log = np.log(a)
        a_log[a <= 0] = 0.0
        return a_log

def log_multivariate_poisson_density(X, means) :
  # # modeled on log_multivariate_normal_density from sklearn.mixture
  n_samples, n_dim = X.shape
  # -lambda + k log(lambda) - log(k!)
  log_means = np.where(means > 1e-3, np.log(means), np.log(1e-3))
  lpr =  np.dot(X, log_means.T)
  lpr = lpr - np.sum(means,axis=1) # rates for all elements are summed and then broadcast across the observation dimenension
  log_factorial = np.sum(gammaln(X + 1), axis=1)
  lpr = lpr - log_factorial[:,None] # logfactobs vector broad cast across the state dimension
  return lpr
=======
        return np.log(a)


def fill_covars(covars, covariance_type='full', n_components=1, n_features=1):
    if covariance_type == 'full':
        return covars
    elif covariance_type == 'diag':
        return np.array(list(map(np.diag, covars)))
    elif covariance_type == 'tied':
        return np.tile(covars, (n_components, 1, 1))
    elif covariance_type == 'spherical':
        eye = np.eye(n_features)[np.newaxis, :, :]
        covars = covars[:, np.newaxis, np.newaxis]
        return eye * covars
>>>>>>> add35ff41052fc23e76fbd484dbed8ddcb1a452d:lib/hmmlearn/utils.py
