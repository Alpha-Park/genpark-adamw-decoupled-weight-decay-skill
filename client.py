import math

class AdamWOptimizer:
    """
    AdamW Optimizer (Decoupled Weight Decay).
    Avoids coupling L2 regularization into exponential moving averages of gradients.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, param, grad, param_id=0):
        self.t += 1
        if param_id not in self.m:
            self.m[param_id] = 0.0
            self.v[param_id] = 0.0

        # Update biased 1st and 2nd moment estimate
        self.m[param_id] = self.b1 * self.m[param_id] + (1.0 - self.b1) * grad
        self.v[param_id] = self.b2 * self.v[param_id] + (1.0 - self.b2) * (grad ** 2)

        # Compute bias-corrected estimates
        m_hat = self.m[param_id] / (1.0 - self.b1 ** self.t)
        v_hat = self.v[param_id] / (1.0 - self.b2 ** self.t)

        # Perform decoupled weight decay
        param = param - self.lr * self.wd * param

        # Update parameter with adaptive step
        param = param - self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        return param
