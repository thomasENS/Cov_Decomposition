# %% Covariance Matrix Decomposition : Toy Example
import numpy as np
from scipy.optimize import minimize

# Candidate covariance matrices
Sigma1 = np.array([[1.0, 0.2], [0.2, 1.5]])
Sigma2 = np.array([[1.5, 0.4], [0.4, 2.5]])
Sigma3 = np.array([[0.8, 0.3], [0.3, 1.2]])
Sigma4 = np.array([[1.0, 0.0], [0.0, 1.0]])

# List of candidate matrices
Sigma_list = [Sigma1, Sigma2, Sigma3, Sigma4]
q = len(Sigma_list)

# Toy example: target covariance matrix
w_true = np.random.random(q); w_true[0:2] = 0 
Sigma  = sum(w_true[j] * Sigma_list[j] for j in range(q))

def objective(w):
    S = sum(w[i] * Sigma_list[i] for i in range(q))
    residual = Sigma - S
    return np.linalg.norm(residual, 'fro')

# Constraints: weights should be non-negative
constraints = [{'type': 'ineq', 'fun': lambda w: w}]

# Initial guess for weights
w0 = np.ones(q) / q

# Optimization
result = minimize(objective, w0, constraints=constraints)

# Optimal weights
w_opt = result.x
S_opt = sum(w_opt[j] * Sigma_list[j] for j in range(q))

print("Optimal Weights:", w_opt)
print("Original Weights:", w_true)
print("Optimal Covariance Matrix:\n", S_opt)
print("Original Covariance Matrix:\n", Sigma)