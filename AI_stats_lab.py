import math
import numpy as np


def bernoulli_log_likelihood(data, theta):
    """
    Compute the Bernoulli log-likelihood for binary data.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    theta : float
        Bernoulli parameter, must satisfy 0 < theta < 1.

    Returns
    -------
    float
        Log-likelihood:
            sum_i [x_i log(theta) + (1-x_i) log(1-theta)]
    """
    data = list(data)

    if len(data) == 0:
        raise ValueError("Data must not be empty.")

    if not (0 < theta < 1):
        raise ValueError("theta must satisfy 0 < theta < 1.")

    for x in data:
        if x not in (0, 1):
            raise ValueError(f"Data must contain only 0s and 1s, got: {x}")

    log_lik = sum(x * math.log(theta) + (1 - x) * math.log(1 - theta) for x in data)
    return log_lik


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    """
    Estimate the Bernoulli MLE and compare candidate theta values.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    candidate_thetas : array-like or None
        Optional candidate theta values to compare using log-likelihood.
        If None, use [0.2, 0.5, 0.8].

    Returns
    -------
    dict
        - 'mle'             : float  — the Bernoulli MLE (sample mean)
        - 'num_successes'   : int
        - 'num_failures'    : int
        - 'log_likelihoods' : dict   — {theta: log-likelihood}
        - 'best_candidate'  : float  — candidate with highest log-likelihood
    """
    data = list(data)

    # Validate via the log-likelihood function with a safe dummy theta
    if len(data) == 0:
        raise ValueError("Data must not be empty.")
    for x in data:
        if x not in (0, 1):
            raise ValueError(f"Data must contain only 0s and 1s, got: {x}")

    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    num_successes = int(sum(data))
    num_failures  = len(data) - num_successes
    mle           = num_successes / len(data)

    log_likelihoods = {}
    for theta in candidate_thetas:
        log_likelihoods[theta] = bernoulli_log_likelihood(data, theta)

    # First-encountered maximum in case of ties
    best_candidate = candidate_thetas[0]
    for theta in candidate_thetas[1:]:
        if log_likelihoods[theta] > log_likelihoods[best_candidate]:
            best_candidate = theta

    return {
        "mle":             mle,
        "num_successes":   num_successes,
        "num_failures":    num_failures,
        "log_likelihoods": log_likelihoods,
        "best_candidate":  best_candidate,
    }


def poisson_log_likelihood(data, lam):
    """
    Compute the Poisson log-likelihood for count data.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    lam : float
        Poisson rate, must satisfy lam > 0.

    Returns
    -------
    float
        Log-likelihood:
            sum_i [x_i log(lam) - lam - log(x_i!)]

    Notes
    -----
    Uses math.lgamma(x + 1) for log(x!) since log(x!) = lgamma(x+1).
    """
    data = list(data)

    if len(data) == 0:
        raise ValueError("Data must not be empty.")

    if lam <= 0:
        raise ValueError("lam must satisfy lam > 0.")

    for x in data:
        if x < 0 or not float(x).is_integer():
            raise ValueError(
                f"Data must contain nonnegative integers, got: {x}"
            )

    log_lik = sum(
        x * math.log(lam) - lam - math.lgamma(x + 1)
        for x in data
    )
    return log_lik


def poisson_mle_analysis(data, candidate_lambdas=None):
    """
    Estimate the Poisson MLE and compare candidate lambda values.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    candidate_lambdas : array-like or None
        Optional candidate lambdas to compare using log-likelihood.
        If None, use [1.0, 3.0, 5.0].

    Returns
    -------
    dict
        - 'mle'             : float — the Poisson MLE (sample mean)
        - 'sample_mean'     : float
        - 'total_count'     : int
        - 'n'               : int
        - 'log_likelihoods' : dict  — {lambda: log-likelihood}
        - 'best_candidate'  : float — candidate with highest log-likelihood
    """
    data = list(data)

    # Validate via the log-likelihood function with a safe dummy lambda
    if len(data) == 0:
        raise ValueError("Data must not be empty.")
    for x in data:
        if x < 0 or not float(x).is_integer():
            raise ValueError(
                f"Data must contain nonnegative integers, got: {x}"
            )

    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    n           = len(data)
    total_count = int(sum(data))
    sample_mean = total_count / n
    mle         = sample_mean          # Poisson MLE = x̄

    log_likelihoods = {}
    for lam in candidate_lambdas:
        log_likelihoods[lam] = poisson_log_likelihood(data, lam)

    # First-encountered maximum in case of ties
    best_candidate = candidate_lambdas[0]
    for lam in candidate_lambdas[1:]:
        if log_likelihoods[lam] > log_likelihoods[best_candidate]:
            best_candidate = lam

    return {
        "mle":             mle,
        "sample_mean":     sample_mean,
        "total_count":     total_count,
        "n":               n,
        "log_likelihoods": log_likelihoods,
        "best_candidate":  best_candidate,
    }


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Bernoulli MLE ===")
    b_data = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]   # 6 successes → MLE = 0.6
    b_res  = bernoulli_mle_with_comparison(b_data, candidate_thetas=[0.2, 0.5, 0.6, 0.8])
    print(f"  MLE            : {b_res['mle']}")
    print(f"  Successes      : {b_res['num_successes']}")
    print(f"  Failures       : {b_res['num_failures']}")
    print(f"  Log-likelihoods: {b_res['log_likelihoods']}")
    print(f"  Best candidate : {b_res['best_candidate']}")

    print()
    print("=== Poisson MLE ===")
    p_data = [2, 3, 4, 2, 3, 5, 1, 4, 3, 3]   # mean = 3.0 → MLE = 3.0
    p_res  = poisson_mle_analysis(p_data, candidate_lambdas=[1.0, 2.0, 3.0, 4.0])
    print(f"  MLE            : {p_res['mle']}")
    print(f"  Sample mean    : {p_res['sample_mean']}")
    print(f"  Total count    : {p_res['total_count']}")
    print(f"  n              : {p_res['n']}")
    print(f"  Log-likelihoods: {p_res['log_likelihoods']}")
    print(f"  Best candidate : {p_res['best_candidate']}")
