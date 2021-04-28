import numpy as np


class ElitistES:
    """
    TODO
    """

    def __init__(self, x0, sigma0, options=None):
        self.x = x0
        self.sigma = sigma0
        self.dim = len(x0)
        self.A = np.eye(self.dim)

        self.tolsig = 1e-5

    def ask(self):
        """
        Sample a candidate solution from x
        """
        return self.x + self.sigma * self.A @ np.random.normal(size=self.dim)

    def tell(self, x, f):
        """
        Update the ES internal model from x and its objective value f(x)
        TODO
        """
        pass

    def _updateStepSize():
        """
        TODO
        """
        pass

    def _updateCholesky():
        """
        TODO
        """
        pass

    def stop(self):
        """
        TODO implement other stopping criteria
        """
        if self.sigma < self.tolsig:
            return True


def fmin(f, x0, sigma0, options=None):
    """
    Standard interface to unconstrained optimization
    """
    es = ElitistES(x0, sigma0, options)
    while not es.stop():
        x = es.ask()
        es.tell(x, f(x))

    return x


def sphere(x):
    x = np.asarray(x)
    return sum(x**2)


if __name__ == "__main__":
    x = fmin(sphere, np.ones(5), 1)

