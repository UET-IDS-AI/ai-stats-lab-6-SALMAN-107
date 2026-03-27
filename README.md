[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/K3Kmbin0)
# 🧪 AI Stats Lab: Maximum Likelihood Estimation (MLE)

## 🎯 Overview

In this lab, you will implement Maximum Likelihood Estimation (MLE) for two important distributions:

- Bernoulli (binary data)
- Poisson (count data)

---

## 🎯 You have TWO tasks

---

# 🔵 Task 1: Bernoulli MLE Analysis

### You must implement:

1. `bernoulli_log_likelihood(data, theta)`  
2. `bernoulli_mle_with_comparison(data, candidate_thetas=None)`

---

## 📌 Model

\[
X_i \in \{0,1\}, \quad P(X=1)=\theta
\]

---

## 🧮 Details

### 1. Log-Likelihood

\[
\ell(\theta) = \sum x_i \log(\theta) + (1-x_i)\log(1-\theta)
\]

### Requirements:
- Data must be non-empty  
- Values must be 0 or 1  
- \(0 < \theta < 1\)  

---

### 2. MLE Analysis Function

`bernoulli_mle_with_comparison(data, candidate_thetas=None)`

This function should:

- Compute the **MLE of θ** using:
  \[
  \hat{\theta} = \frac{1}{n} \sum x_i
  \]

- Count:
  - number of successes (1s)
  - number of failures (0s)

- Compute log-likelihood for each candidate θ

---

### Return a dictionary containing:

- `mle` → estimated θ  
- `num_successes`  
- `num_failures`  
- `log_likelihoods` → {θ: log-likelihood}  
- `best_candidate` → θ with highest likelihood  

---

## 💡 Intuition

- Bernoulli models **binary outcomes**  
- MLE = **observed proportion of successes**

---

# 🔴 Task 2: Poisson MLE Analysis

### You must implement:

1. `poisson_log_likelihood(data, lam)`  
2. `poisson_mle_analysis(data, candidate_lambdas=None)`

---

## 📌 Model

\[
X_i \in \{0,1,2,\dots\}, \quad X_i \sim \text{Poisson}(\lambda)
\]

---

## 🧮 Details

### 1. Log-Likelihood

\[
\ell(\lambda) = \sum \left[x_i \log(\lambda) - \lambda - \log(x_i!)\right]
\]

### Requirements:
- Data must be non-empty  
- Values must be nonnegative integers  
- \(\lambda > 0\)  

### Hint:
```python
math.lgamma(x + 1)
