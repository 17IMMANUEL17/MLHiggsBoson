import numpy as np

from core.costs import sigmoid, log_likelihood_loss


def evaluate_armijo_rule(f_x, f_x1, p, grad, c, alpha):
    """
       Check the armijo rule, return true if the armijo condition is satisfied

        Input:
        - f_x : float
            The function value at x
        - f_x1 : float
            The function value at x_(k+1) = x_k + alpha * p_k
        - p: np.array(D, 1)
            The flatten search direction
        - grad: np.array(D, 1)
            The gradient of the function at x
        - c: float
            The coefficient for armijo condition
        - alpha: float
            The current step size

        Output:
        - condition: bool
            True if the armijio condition is satisfied
    """

    # rule: f(x + alpha * p) <= f(x) + c * alpha * p^T * grad(x)
    return f_x1 <= f_x + c * alpha * (p.reshape(-1, 1).T @ grad.reshape(-1, 1)).item()


def backtracking_line_search(p, grad, w, gamma, beta, c, f, *arg):
    """
        Computes the step size for p that satisfies the armijio condition.

        Input:
        - p: np.array(D, 1)
            The flatten search direction
        - grad: np.array(D, 1)
            The gradient of the function at w
        - w : np.array(D, 1)
            The array of optimization variables, weights
        - gamma: float
            The initial step size
        - beta : float
            The backtracking ratio, alpha = beta * alpha
        - c: float
            The coefficient for armijo condition
        - f: function
            The objective function (optimization_objective, loss)
        - *arg: parameters
            The rest parameters for the function f except the its first variables Vx

        Output:
        - alpha: float
           The step size for p that satisfies the armijio condition
    """

    alpha = gamma
    w_new = w.copy()
    w_new = w + alpha * p
    f_x = f(w, *arg)
    while not evaluate_armijo_rule(f_x, f(w_new, *arg), 
                                    p, grad, c, alpha):
        alpha = beta * alpha
        w_new = w + alpha * p
    return alpha


def LR_compute_gradient(tx, y_true, y_pred):
    # This is implemented here as well to avoid circular imports
    # when we put bfgs in the implementations.py file, we can remove this
    """
    Compute the gradient.
    Args:
        tx: numpy array of shape (N,D), D is the number of features.
        y_true: true labels, numpy array of shape (N,)
        y_pred: predicted labels, numpy array of shape (N,)
    Returns:
        grad: gradient, numpy array of shape (D,)
    """
    return tx.T.dot(y_pred - y_true) / len(y_true)


def LR_optimization_objective(w, tx, y):
    y_pred = sigmoid(tx @ w)
    loss = log_likelihood_loss(y, y_pred)
    return loss


class BFGS:
    '''
    BFGS method for optimization

    Parameters:
        dim: int, the dimension of the Hessians
        beta: float, the backtracking ratio
        c: float, the coefficient for armijo condition
    
    Methods:
        step: compute the step size for gradient that satisfies the armijo condition

    '''
    def __init__(self, dim, beta, c):
        self.inv_B = np.identity(dim)
        self.beta = beta
        self.c = c
        self.prev_obj = None

        print("BFGS method initialized")

    def update_inverse_approximate_hessian_matrix(self, sk, yk):
        '''
        Update the inverse approximated hessian matrix

        Parameters:
            s_k : np.array(D, 1)
                s_k = w_{k+1} - w_{k}, the difference in variables at two consecutive iterations.
            y_k : np.array(D, 1)
                y_k = grad(f(w_{k+1})) - grad(f(w_{k})), the difference in gradients at two consecutive iterations.
        '''
        inv_B_new = self.inv_B.copy()
        sty = (sk.T @ yk).item() # scalar
        inv_B_new = self.inv_B +  (sty + (yk.T @ self.inv_B @ yk).item()) / (sty * sty) * (sk @ sk.T) -\
                    (self.inv_B @ yk @ sk.T + sk @ yk.T @ self.inv_B) / sty
        self.inv_B = inv_B_new

    def step(self, w, tx, y, gamma, lambda_=0.0, tol=1e-8):
        '''
        Compute the step size for p that satisfies the armijio condition.
        '''
        w = w.reshape(-1, 1)
        y = y.reshape(-1, 1)

        y_pred = sigmoid(tx.dot(w))
        loss = log_likelihood_loss(y, y_pred)

        if self.prev_obj is not None and abs(loss - self.prev_obj) < tol:
            return w, loss
        self.prev_obj = loss

        grad = LR_compute_gradient(tx, y, y_pred).reshape(-1, 1) + 2 * lambda_ * w
        pk = - self.inv_B @ grad # (D,) 

        alpha = backtracking_line_search(pk, grad, w, gamma, self.beta, self.c, 
                                         LR_optimization_objective, tx, y)
        sk = alpha * pk # (D,)
        w = w + sk

        next_grad = LR_compute_gradient(tx, y, sigmoid(tx.dot(w))) + 2 * lambda_ * w
        yk = (next_grad - grad).reshape(-1, 1) # (D,1)
        self.update_inverse_approximate_hessian_matrix(sk, yk)

        return w, loss
