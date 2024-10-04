import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from itertools import combinations
from scipy.special import comb, erf, gammaln
from scipy.stats import norm
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
from tqdm import tqdm
import warnings

from piglasso import QJSweeper

warnings.simplefilter('error', OptimizeWarning)

def estimate_lambda_np(edge_counts_all, Q, lambda_range):
    p, _, J = edge_counts_all.shape  # Get the dimensions from edge_counts_all

    # Get the indices for the lower triangular part of the matrix, excluding the diagonal
    lower_tri_indices = np.tril_indices(p, -1)

    # edge_counts_all = edge_counts_all / (2 * Q)


    # Extract the lower triangular part for each lambda
    N_k_matrix = np.zeros((p * (p - 1) // 2, J))
    for k in range(J):
        N_k_matrix[:, k] = edge_counts_all[:, :, k][lower_tri_indices]

    # Calculate the empirical probability p_k for each edge for each lambda
    p_k_matrix = N_k_matrix / Q
    p_k_matrix = np.clip(p_k_matrix, 1e-5, 1 - 1e-5)  # Regularize probabilities to avoid 0 or 1

    # Calculate theta_matrix using the lower triangular indices for each lambda
    theta_matrix = np.zeros_like(N_k_matrix)
    for k in range(J):
        edge_counts_lambda = N_k_matrix[:, k]
        log_theta = log_comb(Q, edge_counts_lambda) \
                    + edge_counts_lambda * np.log(p_k_matrix[:, k]) \
                    + (Q - edge_counts_lambda) * np.log(1 - p_k_matrix[:, k])
        theta_matrix[:, k] = np.exp(log_theta)

    # Calculate f_k and g for each edge across all lambda values
    f_k_lj_matrix = N_k_matrix / Q
    g_matrix = 4 * f_k_lj_matrix * (1 - f_k_lj_matrix)

    # Reshape the matrices for vectorized operations
    theta_matrix_reshaped = theta_matrix.reshape(-1, J)
    g_matrix_reshaped = g_matrix.reshape(-1, J)

    # Compute the score for each lambda
    scores = np.sum(theta_matrix_reshaped * (1 - g_matrix_reshaped), axis=0)

    # Find the lambda that maximizes the score
    lambda_np = lambda_range[np.argmax(scores)]

    return lambda_np, theta_matrix

def log_comb(n, k):
    """Compute the logarithm of combinations using gamma logarithm for numerical stability."""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def find_invalid_values(arr):
    if np.any(np.isnan(arr)):
        return "NaN found", arr[np.isnan(arr)]
    if np.any(np.isinf(arr)):
        return "Inf found", arr[np.isinf(arr)]
    return "No invalid values found"


# def estimate_lambda_wp(edge_counts_all, Q, lambda_range, prior_matrix):
#     p, _, J = edge_counts_all.shape  # Shape of edge_counts_all

#     # Lower triangular indices (excluding diagonal)
#     lower_tri_indices = np.tril_indices(p, -1)

#     # Extracting the lower triangular part for each lambda
#     N_k_matrix = np.zeros((p * (p - 1) // 2, J))
#     for k in range(J):
#         N_k_matrix[:, k] = edge_counts_all[:, :, k][lower_tri_indices]  # Shape (p*(p-1)/2, J)

#     # Empirical probability matrix for each edge for each lambda
#     p_k_matrix = N_k_matrix / Q  # Shape (p *(p-1)/2, J)
#     p_k_matrix = np.clip(p_k_matrix, 1e-5, 1 - 1e-5) # Regularizing probabilities

#     # Reshape the prior matrix to contain only the lower triangular part
#     prior_vector = prior_matrix[lower_tri_indices]  # Shape (p*(p-1)/2,)

#     # Count matrix for each edge across lambdas
#     count_mat = N_k_matrix  # Already in the required shape

#     # Calculation of mus and variances for the data distribution
#     mus = p_k_matrix * Q  # Shape (p*(p-1)/2, J)
#     variances = p_k_matrix * (1 - p_k_matrix) * Q  # Shape (p*(p-1)/2, J)
#     variances = np.clip(variances, 1e-10, np.inf)

#     # Prior distribution parameters
#     psis = prior_vector * Q  # Shape (p*(p-1)/2,)
#     tau_tr = np.sum(np.abs(mus - psis[:, None])) / mus.shape[0]  # Scalar
#     tau_tr = np.clip(tau_tr, 1e-10, np.inf)

#     # Posterior distribution parameters
#     post_mus = (mus * tau_tr**2 + psis[:, None] * variances) / (variances + tau_tr**2)  # Shape (p*(p-1)/2, J)
#     post_var = (variances * tau_tr**2) / (variances + tau_tr**2)  # Shape (p*(p-1)/2, J)

#     # Normal distribution CDF calculations
#     epsilon = 1e-5
#     z_scores_plus = (count_mat + epsilon - post_mus) / np.sqrt(post_var)
#     z_scores_minus = (count_mat - epsilon - post_mus) / np.sqrt(post_var)
#     thetas = 0.5 * (erf(z_scores_plus / np.sqrt(2)) - erf(z_scores_minus / np.sqrt(2)))  # Shape (p*(p-1)/2, J)

#     # Scoring function
#     freq_mat = count_mat / Q  # Shape (p*(p-1)/2, J)
#     g_mat = 4 * freq_mat * (1 - freq_mat)  # Shape (p*(p-1)/2, J)
#     scores = np.sum(thetas * (1 - g_mat), axis=0)  # Shape (J,)

#     # Finding the lambda that maximizes the score
#     lambda_wp = lambda_range[np.argmax(scores)]

#     return lambda_wp, tau_tr, post_mus

# OLD LAMBDA WP
def estimate_lambda_wp(edge_counts_all, Q, lambda_range, prior_matrix):
    """
    Estimates the lambda value for the prior edges.
    Parameters
    ----------
    data : array-like, shape (n, p)
        The data matrix.
    b : int
        The size of the sub-samples.
    Q : int
        The number of sub-samples.
    p_k_matrix : array-like, shape (p, p)
        The probability of an edge being present for each edge, calculated across all sub-samples and lambdas.
    edge_counts_all : array-like, shape (p, p, J)
        The edge counts across sub-samples, for a  a certain lambda.
    lambda_range : array-like, shape (J)
        The range of lambda values.
    prior_matrix : array-like, shape (p, p)
        The prior matrix. Used to identify which edges are penalized by lambda_wp.

    Returns
    -------
    lambda_wp : float
        The lambda value for the prior edges.
    tau_tr : float
        The standard deviation of the prior distribution.
    mus : array-like, shape (p, p)
        The mean of the prior distribution.
    """
    # Check for NaN or infinite values in input arrays
    if np.any(np.isnan(edge_counts_all)) or np.any(np.isinf(edge_counts_all)):
        raise ValueError("edge_counts_all contains NaN or infinite values")

    if np.any(np.isnan(prior_matrix)) or np.any(np.isinf(prior_matrix)):
        raise ValueError("prior_matrix contains NaN or infinite values")

    # Ensure Q is not zero or very close to zero
    if Q == 0 or np.isclose(Q, 0):
        raise ValueError("Q is zero or very close to zero, which may lead to division by zero")

    # prior_matrix = prior_matrix * 0.1

    p, _, _ = edge_counts_all.shape
    J = len(lambda_range)

    # edge_counts_all = edge_counts_all / (2 * Q)

    N_k_matrix = np.sum(edge_counts_all, axis=2)
    p_k_matrix = N_k_matrix / (Q * J) # EDGE_DIVIDER

    # reshape the prior matrix to only contain the edges in the lower triangle of the matrix
    wp_tr_idx = [(i, j) for i, j in combinations(range(p), 2) if prior_matrix[i, j] != 0] # THIS SETS THE INDICES FOR ALL VECTORIZED OPERATIONS

    # wp_tr_weights and p_k_vec give the prob of an edge in the prior and the data, respectively
    wp_tr_weights = np.array([prior_matrix[ind[0], ind[1]] for ind in wp_tr_idx])
    p_k_vec = np.array([p_k_matrix[ind[0], ind[1]] for ind in wp_tr_idx])
    for i, p_k in enumerate(p_k_vec):
        if p_k < 1e-5:
            p_k_vec[i] = 1e-5


    count_mat = np.zeros((J, len(wp_tr_idx))) # Stores counts for each edge across lambdas (shape: lambdas x edges)
    for l in range(J):
        count_mat[l,:] =  [edge_counts_all[ind[0], ind[1], l] for ind in wp_tr_idx]

    # # Alternative code for count_mat (=z_mat)
    # wp_tr_rows, wp_tr_cols = zip(*wp_tr)  # Unzip the wp_tr tuples into two separate lists
    # z_mat = zks[wp_tr_rows, wp_tr_cols, np.arange(len(lambda_range))[:, None]]


    ######### DATA DISTRIBUTION #####################################################################
    # calculate mus, vars
    mus = p_k_vec * Q
    variances = p_k_vec * (1 - p_k_vec) * Q

    ######### PRIOR DISTRIBUTION #####################################################################
    #psi (=prior mean)
    psis = wp_tr_weights * Q # expansion to multipe prior sources: add a third dimension of length r, corresponding to the number of prior sources
    # tau_tr (=SD of the prior distribution)
    # print(f' SUM OF PSIS: {np.sum(psis)}')
    # print(f' SUM OF MUS: {np.sum(mus)}')
    # print(f'ERR: {np.sum(np.abs(mus - psis))}')
    # print(f'LEN OF WP_TR_IDX: {len(wp_tr_idx)}')
    tau_tr = np.sum(np.abs(mus - psis)) / len(wp_tr_idx) # NOTE: eq. 12 alternatively, divide by np.sum(np.abs(wp_tr))

    # Ensure that tau_tr is not 0 or very close to 0
    variances = np.clip(variances, 1e-10, np.inf)
    tau_tr = np.clip(tau_tr, 1e-10, np.inf)

    ######## POSTERIOR DISTRIBUTION ######################################################################
    # Vectorized computation of post_mu and post_var
    post_mus = (mus * tau_tr**2 + psis * variances) / (variances + tau_tr**2)
    post_var = (variances * tau_tr**2) / (variances + tau_tr**2)

    # Since the normal distribution parameters are arrays...
    # Compute the CDF values directly using the formula for the normal distribution CDF
    epsilon = 1e-5

    z_scores_plus = (count_mat + epsilon - post_mus[None, :]) / np.sqrt(post_var)[None, :]
    z_scores_minus = (count_mat - epsilon - post_mus[None, :]) / np.sqrt(post_var)[None, :]

    # Ensure the inputs to erf are within a valid range
    z_scores_plus = np.clip(z_scores_plus, -np.inf, np.inf)
    z_scores_minus = np.clip(z_scores_minus, -np.inf, np.inf)

    # Compute CDF values using the error function
    # By subtracting 2 values of the CDF, the 1s cancel
    thetas = 0.5 * (erf(z_scores_plus / np.sqrt(2)) - erf(z_scores_minus / np.sqrt(2)))
    # print('shape of thetas: ', {thetas.shape})

    ######### SCORING #####################################################################
    # Frequency, instability, and score
    freq_mat = count_mat / Q                                       # shape: lambdas x edges
    g_mat = 4 * freq_mat * (1 - freq_mat)

    # Scoring function
    scores = np.sum(thetas * (1 - g_mat), axis=1)
    # print(scores)

    # # print(scores)
    # print(tau_tr)
    # print(np.sum(p_k_vec) / len(p_k_vec))


    # Find the lambda_j that maximizes the score
    lambda_wp = lambda_range[np.argmax(scores)]

    # print(lambda_wp)

    return lambda_wp, tau_tr, mus









# Shapes of variables:
# - edge_counts_all: (p, p, J)
# - Q: scalar
# - lambda_range: (J)
# - prior_matrix: (p, p)
# - wp_tr_idx: List of tuples representing lower triangular indices
# - wp_tr_weights: (r), where r is the number of non-zero entries in the lower triangle of prior_matrix
# - p_k_vec: (r, J)
# - count_mat: (J, r)
# - mus: (r, J)
# - variances: (r, J)
# - psis: (r)
# - tau_tr: scalar
# - post_mus: (r, J)
# - post_var: (r, J)
# - z_scores_plus: (J, r)
# - z_scores_minus: (J, r)
# - thetas: (J, r)
# - freq_mat: (J, r)
# - g_mat: (J, r)
# - scores: (J)
# - lambda_wp: scalar



# Define a linear function for curve fitting
def linear_func(x, a, b):
    return a * x + b

def fit_lines_and_get_error(index, lambdas, edge_counts, left_bound, right_bound):
    # Only consider data points within the specified bounds
    left_data = lambdas[left_bound:index+1]
    right_data = lambdas[index:right_bound]

    if len(left_data) < 10 or len(right_data) < 10:
        return np.inf

    # Fit lines to the left and right of current index within bounds
    # print(index)
    try:
        params_left, _ = curve_fit(linear_func, left_data, edge_counts[left_bound:index+1])
    except:
        print(f'LEFT DATA: problematic curve fit for lambda kneepoints: at lambda index {index}')
        print(f'left indices len: {len(left_data)}')
        params_left = (0,0)
    try:
        params_right, _ = curve_fit(linear_func, right_data, edge_counts[index:right_bound])
    except:
        print(f'RIGHT DATA: problematic curve fit for lambda kneepoints: at lambda index {index}')
        print(f'right indices len: {len(right_data)}')
        params_right = (0,0)

    # Calculate fit errors within bounds
    error_left = np.sum((linear_func(left_data, *params_left) - edge_counts[left_bound:index+1]) ** 2)
    error_right = np.sum((linear_func(right_data, *params_right) - edge_counts[index:right_bound]) ** 2)

    return error_left + error_right

def find_knee_point(lambda_range, edge_counts_all, left_bound, right_bound):
    errors = [fit_lines_and_get_error(i, lambda_range, edge_counts_all, left_bound, right_bound)
              for i in range(left_bound, right_bound)]
    knee_point_index = np.argmin(errors) + left_bound
    return knee_point_index

def find_all_knee_points(lambda_range, edge_counts_all):
    # Sum the edge counts across all nodes
    edge_counts_all = np.sum(edge_counts_all, axis=(0, 1))

    # Find the main knee point across the full range
    main_knee_point_index = find_knee_point(lambda_range, edge_counts_all, 0, len(lambda_range))
    main_knee_point = lambda_range[main_knee_point_index]

    # For the left knee point, consider points to the left of the main knee point
    left_knee_point_index = find_knee_point(lambda_range, edge_counts_all, 0, main_knee_point_index)
    left_knee_point = lambda_range[left_knee_point_index]

    # For the right knee point, consider points to the right of the main knee point
    # Update the bounds to ensure the fit_lines_and_get_error function considers only the right subset
    right_knee_point_index = find_knee_point(lambda_range, edge_counts_all, main_knee_point_index, len(lambda_range))
    right_knee_point = lambda_range[right_knee_point_index]

    return left_knee_point, main_knee_point, right_knee_point, left_knee_point_index, main_knee_point_index, right_knee_point_index

# Main code
if __name__ == "__main__":
    #### Main code ####
    p = 154
    n = 1337 # [50, 100, 200, 400, 750, 1000, 2000]
    b_perc = 0.65
    b = int(b_perc * n)   # size of sub-samples
    Q = 1200          # number of sub-samples

    lowerbound = 0.01
    upperbound = 0.9
    granularity = 300
    lambda_range = np.linspace(lowerbound, upperbound, granularity)

    fp_fn = 0
    skew = 0
    density = 0.03
    seed = 42

    omics_type =  'proteomics' # 'synthetic'
    cms = 'cmsALL'

    filename_edges = f'Networks/net_results/{omics_type}_{cms}_edge_counts_all_pnQ{p}_{n}_{Q}_{lowerbound}_{upperbound}_ll{granularity}_b{b_perc}_fpfn{fp_fn}_skew{skew}_dens{density}_s{seed}.pkl'
    with open(filename_edges, 'rb') as f:
        edge_counts_all = pickle.load(f)


    left_knee_point, main_knee_point, right_knee_point, left_knee_point_index, knee_point_index, right_knee_point_index = find_all_knee_points(lambda_range, edge_counts_all)
    print("Left Knee Point at lambda =", left_knee_point)
    print("Main Knee Point at lambda =", main_knee_point)
    print("Right Knee Point at lambda =", right_knee_point)

    # We will now plot the additional lines: the right red line and the left magenta line
    # Sum the edge counts across all nodes
    edge_counts_all = np.sum(edge_counts_all, axis=(0, 1))

    window_size = 100
    def smooth_data(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    edge_counts_all = smooth_data(edge_counts_all, window_size)
    lambda_range = lambda_range[:len(edge_counts_all)]

    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(lambda_range, edge_counts_all, color='grey', label='Edge Counts', alpha = 0.4)

    # Fit and plot the lines for the left knee point
    left_data = lambda_range[:left_knee_point_index+1]
    left_fit_params, _ = curve_fit(linear_func, left_data, edge_counts_all[:left_knee_point_index+1])
    plt.plot(left_data, linear_func(left_data, *left_fit_params), 'r-', label='Left Fit')

    # Fit and plot the line between the left knee point and the main knee point (right red line)
    left_knee_to_main_data = lambda_range[left_knee_point_index:knee_point_index+1]
    left_knee_to_main_fit_params, _ = curve_fit(linear_func, left_knee_to_main_data, edge_counts_all[left_knee_point_index:knee_point_index+1])
    plt.plot(left_knee_to_main_data, linear_func(left_knee_to_main_data, *left_knee_to_main_fit_params), 'r-')

    # Fit and plot the lines for the main knee point
    main_left_data = lambda_range[:knee_point_index]
    main_right_data = lambda_range[knee_point_index:]
    main_left_fit_params, _ = curve_fit(linear_func, main_left_data, edge_counts_all[:knee_point_index])
    main_right_fit_params, _ = curve_fit(linear_func, main_right_data, edge_counts_all[knee_point_index:])
    plt.plot(main_left_data, linear_func(main_left_data, *main_left_fit_params), 'g-', label='Main Fit')
    plt.plot(main_right_data, linear_func(main_right_data, *main_right_fit_params), 'g-')

    # Fit and plot the line between the main knee point and the right knee point (left magenta line)
    main_to_right_knee_data = lambda_range[knee_point_index:right_knee_point_index+1]
    main_to_right_knee_fit_params, _ = curve_fit(linear_func, main_to_right_knee_data, edge_counts_all[knee_point_index:right_knee_point_index+1])
    plt.plot(main_to_right_knee_data, linear_func(main_to_right_knee_data, *main_to_right_knee_fit_params), 'm-')

    # Fit and plot the lines for the right knee point
    right_data = lambda_range[right_knee_point_index:]
    right_fit_params, _ = curve_fit(linear_func, right_data, edge_counts_all[right_knee_point_index:])
    plt.plot(right_data, linear_func(right_data, *right_fit_params), 'm-', label='Right Fit')

    # Mark the knee points on the plot
    plt.axvline(x=left_knee_point, color='r', linestyle='--', label='Left Knee Point', alpha = 0.5)
    plt.axvline(x=main_knee_point, color='g', linestyle='--', label='Main Knee Point', alpha = 0.5)
    plt.axvline(x=right_knee_point, color='m', linestyle='--', label='Right Knee Point', alpha = 0.5)

    plt.xlabel(r'$ \lambda$', fontsize=15)
    plt.ylabel('Edge Counts', fontsize=12)
    plt.title('Knee Points and Fitted Lines')
    # plt.ylim(0, 8000)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()



