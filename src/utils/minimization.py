import numpy as np

## Frobenius norm objective function
def objective_function(w, S, Sigma_list):
    S_approx = sum(w[j] * Sigma_list[j] for j in range(len(Sigma_list)))
    residual = S - S_approx
    return np.linalg.norm(residual, 'fro')

# Negative Log-likelihood function
def log_likelihood_objective(w, X, Sigma_list):
    S = sum(w[j] * Sigma_list[j] for j in range(q))
    inv_S = np.linalg.inv(S)
    term1 = -n * p / 2 * np.log(2 * np.pi)
    term2 = -n / 2 * np.log(np.linalg.det(S))
    term3 = -0.5 * np.trace(X.T @ X @ inv_S)
    return -(term1 + term2 + term3)  # Negative for minimization

## Constraints: weights must be non-negative
constraints = [{'type': 'ineq', 'fun': lambda w: w}]

## Explained Variance for the given problem
matrix_r2_score = lambda M_pred, M_true: 1 - np.linalg.norm(M_true - M_pred, 'fro')**2 / np.linalg.norm(M_true, 'fro')**2
