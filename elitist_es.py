#!/usr/bin/env python
# coding: utf-8
# In[1]:


import numpy as np
from problem import SphereLinCons
from scipy.stats import ortho_group


class CholeskyElitistES:
    """
    Implementation of the (1+1)-Cholesky-CMA-ES without constraint.
    It is the implementation presented in the article:
    'A Computational Efficient Covariance Matrix Update and a (1+1)CMA for
    Evolution Strategies'
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
            self.stagnation = 0
            self.fct.append(f)
            self.best.append(f)
            self._updateCholesky()

        else:
            self.fct.append(self.fct[-1])
            self.stagnation += 1

    def _updateStepSize(self, lbd):
        """
        Update the value of the step size sigma and the averaged success rate,
        p_succ.
        """
        self.p_succ = (1 - self.c_p) * self.p_succ + self.c_p * lbd
        self.sigma *= np.exp(1/self.d * ((self.p_succ - self.p_target_succ)
                                         / (1 - self.p_target_succ)))

    def _updateCholesky(self):
        """
        Update of the cholesky matrix in order to change the search space for
        new candidates
        """
        if self.p_succ < self.p_thresh:
            c_a = np.sqrt(1 - self.c_cov)
            update_coef = c_a / np.linalg.norm(self.z) \
                * (np.sqrt(1 + (1 - c_a**2) * np.linalg.norm(self.z)**2
                           / c_a**2) - 1)
            self.A *= c_a + update_coef * np.outer(self.z, self.z)

    def stop(self):
        """
        Stopping criteria
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
    es = CholeskyElitistES(x0, sigma0, options)
    while not es.stop():
        x = es.ask()
        es.tell(x, f(x))

    return x


class ActiveElitistES:
    """
    It is the implementation of the algorithm presented in the article:
    'A (1+1)-CMA-ES for Constrained Optimisation'
    by D. V. Arnold, and N. Hansen.
    """

    def __init__(self, x0, sigma0, options=None):
        self.x = x0
        self.sigma = sigma0
        self.dim = len(x0)
        self.A = np.eye(self.dim)
        self.fct = [1e15]
        self.z = np.ones(self.dim) * 1e-4
        self.v = np.array([])
        self.w = np.array([])

        # Parameter settings
        self.d = 1 + self.dim/2  # the damping parameter which controls the rate of the step size adaptation
        self.c = 2/(self.dim+2)
        self.c_p = 1/12  # learning rate of the average success
        self.p_target = 2/11  # target succes rate
        self.c_cov_plus = 2/(self.dim**2 + 6)
        self.c_c = 1/(self.dim+2)
        self.beta = 0.1/(self.dim+2)

        # Variable:
        self.p_succ = 2/11  # p_target
        self.fifth_order = np.ones(5) * np.inf
        self.s = 0

        # Parameters for stopping criterium :
        self.tolsig = 1e-12
        self.stagnation = 0
        self.best = []
        self.TolX = 1e-12 * sigma0

        self.stop_now = False

    def ask(self):
        """
        Sample a candidate solution from x
        """
        self.z = np.random.normal(size=self.dim)

        return self.x + self.sigma * self.A.dot(self.z)

    def tell(self, x_new, objective, g):
        """
        Update the ES internal model from x and its objective value f(x)
        """
        not_good = self._updateConstraint(g)
        if not not_good:
            f = objective(x_new)
            lbd = 1 * (f <= self.fct[-1])
            self._updateStepSize(lbd)

            if lbd == 1:
                self.x = x_new
                self.fct.append(f)
                self.best.append(f)
                self._updateCholesky()
                self.stagnation = 0

            else:
                self.fct.append(self.fct[-1])
                self.stagnation += 1

            if len(self.fct) > 5 and sum(self.fifth_order > f) == 0:
                self._updateFifthOrder()
            self.fifth_order = np.concatenate([self.fifth_order[1:], [f]])

    def _updateConstraint(self, g):
        """
        If the solution isn't feasible we update the cholesky matrix, A and the
        exponentially fading record, v.
        """
        m = len(g)
        infeasible = False
        summ = 0

        # Init
        if self.v.shape == (0,):
            self.v = np.zeros((m, self.dim))
            self.w = np.zeros((m, self.dim))

        # We take the inverse of A only if the solution is infeasible
        if sum(g > 0) > 0:
            inv_A = np.linalg.inv(self.A)
            infeasible = True

        for j in range(m):
            if g[j] > 0:
                self.v[j] *= (1-self.c_c)
                self.v[j] += self.c_c * self.A.dot(self.z)
                self.w[j] = inv_A.dot(self.v[j])
                summ += np.outer(self.v[j], self.w[j]) / self.w[j].T.dot(self.w[j])

        if infeasible:
            self.A -= self.beta / np.sum(g > 0) * summ

        return infeasible

    def _updateStepSize(self, lbd):
        """
        Update the value of the step size sigma and the averaged success rate,
        p_succ.
        """
        self.p_succ = (1-self.c_p) * self.p_succ + self.c_p * lbd
        self.sigma *= np.exp((self.p_succ - 2.0/11.0) / ((1.0-2.0/11.0)*self.d))

    def _updateCholesky(self):
        """
        Update of the cholesky matrix and the exponentially fading record, s,
        in order to change the search space for new candidates.
        Rather than working with the covariance matrix and performing a
        Cholesky decompositionin every iteration of the algorithm, Igel et al.
        presented a direct update of A.
        """
        self.s *= (1-self.c)
        self.s += np.sqrt(self.c * (2-self.c)) * self.A.dot(self.z)
        w2 = np.linalg.inv(self.A).dot(self.s)
        self.A *= np.sqrt(1 - self.c_cov_plus)
        self.A += np.sqrt(1 - self.c_cov_plus) / np.linalg.norm(w2)**2 \
            * (np.sqrt(1 + self.c_cov_plus * np.linalg.norm(w2)**2
                       / (1-self.c_cov_plus)) - 1) * np.outer(self.s, w2)

    def _updateFifthOrder(self):
        """
        In the case where the  solution is worst than the fifth last, we
        incorporate the active covariance matrix update due to Jastrebski and
        Arnold.
        """
        self.c_cov_minus = np.min([0.4/(self.dim**(1.6) + 1),
                                   1/(2*np.linalg.norm(self.z)**2 - 1)])
        self.A *= np.sqrt(1 + self.c_cov_minus)
        self.A += np.sqrt(1 + self.c_cov_minus) / np.linalg.norm(self.z)**2 \
            * (np.sqrt(1 - self.c_cov_minus * np.linalg.norm(self.z)**2
                       / (1+self.c_cov_minus)) - 1) \
            * self.A.dot(np.outer(self.z, self.z))

    def stop(self):
        """
        Stopping criteria
        """
        if self.sigma < self.tolsig:
            print("sigma")
            return True
        elif self.stagnation > 120 + 30*self.dim:
            # Stagnation crit
            print("Stagnation crit")
            return True
        elif len(self.best) > 2 and self.best[-2] - self.best[-1] < 1e-16:
            # TolFun crit
            print("TolFun crit")
            return True
        elif self.sigma * self.p_succ < self.TolX:
            # TolX crit
            print("TolX crit")
            return True

    def stopping_criterion(self, f_, g_, minf):
        g = [np.max([u, 0]) for u in g_]
        # stop_bool = np.allclose(g_, 0, atol=self.stop_tol["g"])
        stop_bool = sum(g) == 0
        stop_bool *= np.allclose(f_, minf, rtol=1e-8)
        # stop_bool *= np.abs(f_ - minf) < 1e-8
        self.stop_now = stop_bool

        if self.stagnation > 120 + 30*self.dim:
            print("Stagnation crit")
            return True
        return self.stop_now


def fmin_con(f, g, x0, sigma0, options=None):
    """
    Interface for constrained optimization
    """
    es = ActiveElitistES(x0, sigma0, options)
    while not es.stop():
        x = es.ask()
        es.tell(x, f, g(x))
    return es.x


def fmin_pb(pb, x0, sigma0, options=None):
    """
    Interface to stop when abs(f_star - f) < 1e-8
    """
    es = ActiveElitistES(x0, sigma0, options)
    g_ = pb.g(x0)
    while not es.stopping_criterion(es.fct[-1], g_, pb.solution):
        x = es.ask()
        g_ = pb.g(x)
        es.tell(x, pb.f, g_)

    return es.x


if __name__ == "__main__":
    dimension = 10
    x0 = np.ones(dimension) * dimension
    sigma0 = 1

    problem = SphereLinCons(dimension, 0)
    x = fmin(problem.f, x0, sigma0)
    print(problem)
    print(x)

    problem = SphereLinCons(dimension, int(dimension/2))
    # problem = TR2()
    O = ortho_group.rvs(dimension)
    # x = fmin_con(lambda x: problem.f(O.dot(x)), problem.g, x0, 1)
    x = fmin_pb(problem, x0, sigma0)
    # x = fmin_pb(problem, np.ones(2)*10, 1)
    print(problem)
    print(x)
