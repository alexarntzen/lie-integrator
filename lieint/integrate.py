import numpy as np
import scipy.linalg as sl


def lie_euler(F: callable, y_0, t_0, t_f, h, exp=sl.expm):
    # store time steps
    N = int((t_f - t_0) / h)
    T = np.linspace(t_0, N * h + t_0, N + 1)

    # store solution
    y_dim = y_0.shape
    y = np.zeros((N + 1, *y_dim))
    y[0] = y_0
    for n in range(1, N + 1):
        y[n] = exp(h * F(y[n - 1])) @ y[n - 1]

    return y, T


def com_free_3(F: callable, y_0, t_0, t_f, h, exp=sl.expm):
    # store time steps
    N = int((t_f - t_0) / h)
    T = np.linspace(t_0, N * h + t_0, N + 1)

    # store solution
    y_dim = y_0.shape
    y = np.zeros((N + 1, *y_dim))
    y[0] = y_0
    for n in range(N):
        Y_1 = y[n]
        Y_2 = exp(1 / 3 * h * F(Y_1)) @ Y_1
        Y_3 = exp(2 / 3 * h * F(Y_2)) @ Y_1
        y[n + 1] = exp(-1 / 12 * h * F(Y_1) + 3 / 4 * h * F(Y_3)) @ Y_2
    return y, T


def crouch_grossmann_3(F: callable, y_0, t_0, t_f, h, exp: callable = sl.expm):
    # store time steps
    N = int((t_f - t_0) / h)
    T = np.linspace(t_0, N * h + t_0, N + 1)

    # store solution
    y_dim = y_0.shape
    y = np.zeros((N + 1, *y_dim))
    y[0] = y_0
    for n in range(N):
        Y_1 = y[n]
        Y_2 = exp(1 / 24 * h * F(Y_1)) @ Y_1
        Y_3 = exp(-6 * h * F(Y_2)) @ exp(161 / 24 * h * F(Y_1)) @ Y_1
        y[n + 1] = exp(2 / 3 * h * F(Y_3)) @ exp(-2 / 3 * h * F(Y_2)) @ exp(h * F(Y_1)) @ Y_1
    return y, T


def rkmk_3(F: callable, y_0, t_0, t_f, h, exp: callable = sl.expm):
    # store time steps
    N = int((t_f - t_0) / h)
    T = np.linspace(t_0, N * h + t_0, N + 1)

    # store solution
    y_dim = y_0.shape
    y = np.zeros((N + 1, *y_dim))
    y[0] = y_0
    for n in range(N):
        Y_1 = y[n]
        K_1 = F(Y_1)
        K_1_hat = K_1
        sigma_2 = h / 3 * K_1
        K_2 = F(exp(sigma_2) @ Y_1)
        K_2_hat = K_2 - 0.5 * bracket(sigma_2, K_2) + 1 / 12 * bracket(sigma_2, bracket(sigma_2, K_2))
        sigma_3 = 2 / 3 * h * K_2_hat
        K_3 = F(exp(sigma_3) @ Y_1)
        K_3_hat = K_3 - 0.5 * bracket(sigma_3, K_3) + 1 / 12 * bracket(sigma_3, bracket(sigma_3, K_3))
        sigma_hat = h / 4 * K_1_hat + 3 / 4 * h * K_3_hat

        y[n + 1] = exp(sigma_hat) @ Y_1
    return y, T


def bracket(A, B):
    return A @ B - B @ A


def so3_exp(A: np.ndarray):
    alpha = np.sqrt(0.5 * np.sum(A ** 2))
    n, _ = A.shape
    return np.eye(n) + np.sin(alpha) / alpha * A + (1 - np.cos(alpha)) / alpha ** 2 * A @ A
