"""
TODO
To be used with NLCCO
To run the algorithm on the benchmark
"""
import numpy as np
from elitist_es import ActiveElitistES
from nlcco.base import BaseRunner


class MyRunner(BaseRunner):

    def g_b(self, x):
        return np.concatenate([self.constraint(x), self.problem.g_bounds(x)])

    def stop(self, f):
        if self.es.stagnation > 120 + 30 * self.es.dim:
            print("Stagnation criterion is reached before the true minimum")
            return True
        # return np.allclose(f, problem.fmin, rtol=1e-8)
        return np.abs(f - self.problem.fmin) < 1e-8

    def run(self, x0, sigma0):
        self.es = ActiveElitistES(x0, sigma0)
        f = np.inf
        self.list_sigma = []
        self.A_norm = []
        self.Q_vp = []
        self.list_x = []
        while not self.stop(f):
            while True:
                # Logger:
                self.Q_vp.append(np.linalg.eig(self.es.A.T.dot(self.es.A))[0])
                self.list_sigma.append(self.es.sigma)
                self.A_norm.append(np.linalg.norm(self.es.A))
                self.list_x.append(self.es.x)

                x = self.es.ask()
                g = self.g_b(x)
                is_feasible = self.es.test(g)

                if self.countg % 5000 == 0:
                    print("{0} evaluation of f and {1} of the constraint."
                          .format(self.countf, self.countg))

                if is_feasible:
                    break
            f = self.objective(x)
            self.es.tell(x, f)

        print("We obtain f={0} after {1} evaluation of f and {2} of g."
              .format(self.es.fct[-1], self.countf, self.countg))
        print("The minimizer is:", self.es.x)
