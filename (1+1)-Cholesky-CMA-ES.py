import numpy as np


class Cholesky_11_ES:
    """
    Implementation of the (1+1)-Cholesky-CMA-ES without constraint.

    It is the implementation presented in the article:
    'A Computational Efficient Covariance Matrix Update and a (1+1)CMA for Evolution Strategies'
    by C. Igel, T. Suttorp, and N. Hansen.
    """

    def __init__(self, x0, sigma0, options=None):
        self.x = x0
        self.sigma = sigma0
        self.dim = len(x0)
        self.A = np.eye(self.dim)
        self.fct = [1e15]

        # Parameters for updateStepSize:
        self.p_target_succ = 2 / 11  # target succes rate
        self.p_succ = self.p_target_succ
        self.d = 1 + self.dim / 2  # the damping parameter which controls the rate of the step size adaptation
        self.c_p = 1 / 12  # learning rate of the average success

        # Parameters for updateCholesky:
        self.p_thresh = 0.44
        self.c_cov = 2 / (self.dim**2 + 6)

        # Parameters for stopping criterium :
        self.tolsig = 1e-8
        self.stagnation = 0
        self.best = []
        self.TolX = 1e-12 * sigma0

    def ask(self):
        """
        Sample a candidate solution from x
        """
        self.z = np.random.normal(size=self.dim)
        return self.x + self.sigma * self.A.dot(self.z)

    def tell(self, x, f):
        """
        Update the ES internal model from x and its objective value f(x)
        """
        lbd = 1 * (f <= self.fct[-1])
        self._updateStepSize(lbd)

        if lbd == 1:
            self.x = x
            self.fct.append(f)
            self.best.append(f)
            self._updateCholesky()

        else:
            self.fct.append(self.fct[-1])
            self.stagnation += 1

    def _updateStepSize(self, lbd):
        """
        Update the value of the step size sigma and the averaged success rate, p_succ.
        """
        self.p_succ = (1 - self.c_p) * self.p_succ + self.c_p * lbd
        self.sigma *= np.exp(1 / self.d * (self.p_succ - self.p_target_succ / (1 - self.p_target_succ) * (1 - self.p_succ)))

    def _updateCholesky(self):
        """
        Update of the cholesky matrix in order to change the search space for new candidates
        """
        if self.p_succ < self.p_thresh:
            c_a = np.sqrt(1 - self.c_cov)
            update_coef = c_a / np.linalg.norm(self.z) * (np.sqrt(1 + (1 - c_a**2) * np.linalg.norm(self.z)**2 / c_a**2) - 1)
            self.A = c_a * self.A + update_coef * self.A * np.outer(self.z, self.z)

    def stop(self):
        """
        TODO implement other stopping criteria
        """
        if self.sigma < self.tolsig:
            print("sigma")
            return True
        elif self.stagnation > 120 + 30 * self.dim:
            print("Stagnation crit")
            return True
        elif len(self.best) > 2 and self.best[-2] - self.best[-1] < 1e-12:
            print("TolFun crit")
            return True
        elif self.sigma * self.p_succ < self.TolX:
            print("TolX crit")
            return True


def fmin(f, x0, sigma0, options=None):
    """
    Standard interface to unconstrained optimization
    """
    es = Cholesky_11_ES(x0, sigma0, options)
    while not es.stop():
        x = es.ask()
        es.tell(x, f(x))

    return x


def sphere(x):
    x = np.asarray(x)
    return sum(x**2)


if __name__ == "__main__":
    x = fmin(sphere, np.ones(5), 1)
