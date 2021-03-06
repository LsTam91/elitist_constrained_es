import numpy as np


class CholeskyElitistES:
    """
    Implementation of the (1+1)-Cholesky-CMA-ES without constraint.
    It is the implementation presented in the article:
    'A Computational Efficient Covariance Matrix Update and a (1+1)CMA for
    Evolution Strategies'
    by C. Igel, T. Suttorp, and N. Hansen.
    """

    def __init__(self, x0, sigma0):
        self.x = x0
        self.sigma = sigma0
        self.dim = len(x0)
        self.A = np.eye(self.dim)
        self.fct = [1e15]

        # Parameters for updateStepSize:
        self.p_target_succ = 2 / 11  # target succes rate
        self.p_succ = self.p_target_succ
        self.d = 1 + self.dim / 2  # controls the rate of the step size adaptation
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
        lbd = f <= self.fct[-1]
        self._updateStepSize(lbd)

        if lbd:
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
            update_coef = c_a / np.linalg.norm(self.z)**2 \
                * (np.sqrt(1 + (1 - c_a**2) * np.linalg.norm(self.z)**2
                           / c_a**2) - 1)
            self.A = c_a * self.A + update_coef * self.A.dot(np.outer(self.z, self.z))

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


def fmin(f, x0, sigma0, plot=False):
    """
    Standard interface to unconstrained optimization
    """
    es = CholeskyElitistES(x0, sigma0)
    sig = []
    vp = []
    std = []
    xs = []
    while not es.stop():
        x = es.ask()
        es.tell(x, f(x))
        vp.append(np.linalg.eig(es.A.T.dot(es.A))[0])
        sig.append(es.sigma)
        std.append(np.diag(es.A.dot(es.A)))
        xs.append(es.x)
    if plot:
        return es, vp, sig, std, xs

    return es


class ActiveElitistES:
    """
    It is the implementation of the algorithm presented in the article:
    'A (1+1)-CMA-ES for Constrained Optimisation'
    by D. V. Arnold, and N. Hansen.
    """

    def __init__(self, x0, sigma0,
                 tolsig=1e-10, tolfun=1e-9, TolX=1e-10
                 ):

        # Optimization variables
        self.x = x0
        self.sigma = sigma0
        self.dim = len(x0)
        self.A = np.eye(self.dim)
        self.fct = [1e15]
        self.z = np.ones(self.dim) * 1e-4
        self.v = np.array([])
        self.w = np.array([])

        # Solver variables
        self.count_f = 0
        self.count_g = 0

        # Parameter settings
        self.d = 1 + self.dim / 2  # controls the rate of the step size adaptation
        self.c = 2 / (self.dim + 2)
        self.c_p = 1 / 12  # learning rate of the average success
        self.p_target = 2 / 11  # target succes rate
        self.c_cov_plus = 2 / (self.dim**2 + 6)
        self.c_c = 1 / (self.dim + 2)
        self.beta = 0.1 / (self.dim + 2)

        # Variable:
        self.p_succ = 2 / 11  # p_target
        self.fifth_order = np.ones(5) * np.inf
        self.s = 0

        # Parameters for stopping criterion :
        self.tolsig = tolsig
        self.tolfun = tolfun
        self.stagnation = 0
        self.tolstagnation = 120 + 30 * self.dim
        self.best = []
        self.TolX = TolX * sigma0
        self.tolcountf = np.inf
        self.tolcountg = np.inf

        self.stop_now = False

    def ask(self):
        """
        Sample a candidate solution from x
        """
        self.z = np.random.normal(size=self.dim)

        return self.x + self.sigma * self.A.dot(self.z)

    def tell(self, x_new, f):
        """
        Update the ES internal model from x and its objective value f(x)
        """
        lbd = f <= self.fct[-1]
        self._updateStepSize(lbd)

        if lbd:
            self.x = x_new
            self.fct.append(f)
            self.best.append(f)
            self._updateCholesky()
            self.stagnation = 0

        else:
            self.fct.append(self.fct[-1])
            self.stagnation += 1

        # if len(self.fct) > 5 and sum(self.fifth_order > f) == 0:
        if (not lbd) and self.p_succ < 0.44 and all(self.fifth_order < f):
            self._updateFifthOrder()
        self.fifth_order = np.concatenate([self.fifth_order[1:], [f]])

        self.count_f += 1

    def test(self, g):
        """
        If the solution isn't feasible we update the cholesky matrix, A and the
        exponentially fading record, v.
        """
        m = len(g)
        feasible = True
        summ = 0

        # Init
        if self.v.shape == (0,):
            self.v = np.zeros((m, self.dim))
        if self.w.shape == (0,):
            self.w = np.zeros((m, self.dim))

        # We take the inverse of A only if the solution is infeasible
        if any(u > 0 for u in g):
            inv_A = np.linalg.inv(self.A)
            feasible = False

        for j in range(m):
            if g[j] > 0:
                self.v[j] *= (1 - self.c_c)
                self.v[j] += self.c_c * self.A.dot(self.z)
                self.w[j] = inv_A.dot(self.v[j])
                summ += np.outer(self.v[j], self.w[j]) / self.w[j].T.dot(self.w[j])

        if not feasible:
            self.A -= self.beta / np.sum([u > 0 for u in g]) * summ
            if np.isnan(self.A).any():
                print("ERROR: NaN values in the covariance matrix")
                print(f"After {self.count_g} constraint evaluations")
                print(f"summ value: {summ}")
                raise RuntimeError("NaN values in the covariance matrix")

        self.count_g += 1

        return feasible

    def _updateStepSize(self, lbd):
        """
        Update the value of the step size sigma and the averaged success rate,
        p_succ.
        """
        self.p_succ = (1 - self.c_p) * self.p_succ + self.c_p * lbd
        self.sigma *= np.exp(
            (self.p_succ - self.p_target) / ((1 - self.p_target) * self.d)
        )

    def _updateCholesky(self):
        """
        Update of the cholesky matrix and the exponentially fading record, s,
        in order to change the search space for new candidates.
        Rather than working with the covariance matrix and performing a
        Cholesky decompositionin every iteration of the algorithm, Igel et al.
        presented a direct update of A.
        """
        if self.p_succ < 0.44:
            self.s *= (1 - self.c)
            self.s += np.sqrt(self.c * (2 - self.c)) * self.A.dot(self.z)
            self.alpha = 1 - self.c_cov_plus

        else:
            self.s *= 1 - self.c
            self.alpha = 1 - self.c_cov_plus + self.c_cov_plus * self.c * (2 - self.c)

        u = np.linalg.inv(self.A).dot(self.s)
        u2 = np.linalg.norm(u)**2

        self.A *= np.sqrt(self.alpha)
        self.A += np.sqrt(self.alpha) * (np.sqrt(1 + self.c_cov_plus * u2
                                                 / (self.alpha)) - 1) * np.outer(self.s, u) / u2

        assert not np.isnan(self.A).any()

    def _updateFifthOrder(self):
        """
        In the case where the  solution is worst than the fifth last, we
        incorporate the active covariance matrix update due to Jastrebski and
        Arnold.
        """
        self.c_cov_minus = np.min(
            [
                0.4 / (self.dim**(1.6) + 1),
                1 / abs(2 * np.linalg.norm(self.z)**2 - 1)
            ]
        )
        z2 = np.linalg.norm(self.z)**2

        self.A *= np.sqrt(1 + self.c_cov_minus)
        self.A += np.sqrt(1 + self.c_cov_minus) / z2 \
            * (np.sqrt(1 - self.c_cov_minus * z2 / (1 + self.c_cov_minus)) - 1) \
            * self.A.dot(np.outer(self.z, self.z))
        assert not np.isnan(self.A).any()

    def stop(self, inner=False):
        """
        Stopping criteria
        Set inner to true to test only for criteria related to the inner loop
        """
        if not inner:
            if self.sigma < self.tolsig:
                print("sigma")
                return True
            elif self.stagnation > self.tolstagnation:
                # Stagnation crit
                print("Stagnation crit")
                return True
            elif len(self.best) > 2 and self.best[-2] - self.best[-1] < self.tolfun:
                # TolFun crit
                print("TolFun crit")
                return True
            elif self.sigma * self.p_succ < self.TolX:
                # TolX crit
                print("TolX crit")
                return True
        if self.count_f >= self.tolcountf or self.count_g > self.tolcountg:
            print("Number of evals exceeded")
            return True
        return False


class FastActiveElitistES:
    """
    It is the implementation of the algorithm presented in the article:
    'A (1+1)-CMA-ES for Constrained Optimisation' by D. V. Arnold,
    and N. Hansen. I use the fast update of the covariance matrix in order to
    reduce the update complexity from O(n^3) to O(n^2) for the Cholesky
    update and the fifth order update and from O(n^3+mn^2) to O(mn^2) for the
    active constraint handling, which is a new update.
    """

    def __init__(self, x0, sigma0,
                 tolsig=1e-10, tolfun=1e-9, TolX=1e-10
                 ):
        '''
        x0 : Assert the starting point is in the feaseble space!
        sigma0 : initial step size
        tolsig, tolfun, TolX : stopping criteria,
        check pycma documentation for more information.
        '''
        # Optimization variables
        self.x = x0
        self.sigma = sigma0
        self.dim = len(x0)
        self.A = np.eye(self.dim)
        self.invA = np.eye(self.dim)

        self.Abis = np.eye(self.dim)
        self.invAbis = np.eye(self.dim)

        self.fct = [1e15]
        self.z = np.ones(self.dim) * 1e-4
        self.v = np.array([])
        self.w = np.array([])

        # Solver variables
        self.count_f = 0
        self.count_g = 0

        # Parameter settings
        self.d = 1 + self.dim / 2  # controls the rate of the step size adaptation
        self.c = 2 / (self.dim + 2)
        self.c_p = 1 / 12  # learning rate of the average success
        self.p_target = 2 / 11  # target succes rate
        self.c_cov_plus = 2 / (self.dim**2 + 6)
        self.c_c = 1 / (self.dim + 2)
        self.beta = 0.1 / (self.dim + 2)

        # Variable:
        self.p_succ = 2 / 11  # p_target
        self.fifth_order = np.ones(5) * np.inf
        self.s = 0

        # Parameters for stopping criterion :
        self.tolsig = tolsig
        self.tolfun = tolfun
        self.stagnation = 0
        self.tolstagnation = 120 + 30 * self.dim
        self.best = []
        self.TolX = TolX * sigma0
        self.tolcountf = np.inf
        self.tolcountg = np.inf

        self.stop_now = False

    def ask(self):
        """
        Sample a candidate solution from x
        """
        self.z = np.random.normal(size=self.dim)

        return self.x + self.sigma * self.A.dot(self.z)

    def tell(self, x_new, f):
        """
        Update the ES internal model from x and its objective value f(x)
        """
        lbd = f <= self.fct[-1]
        self._updateStepSize(lbd)

        if lbd:
            self.x = x_new
            self.fct.append(f)
            self.best.append(f)
            self._updateCholesky()
            self.stagnation = 0

        else:
            self.fct.append(self.fct[-1])
            self.stagnation += 1

        if (not lbd) and self.p_succ < 0.44 and all(self.fifth_order < f):
            self._updateFifthOrder()
        self.fifth_order = np.concatenate([self.fifth_order[1:], [f]])

        self.count_f += 1

    def test(self, g):
        """
        If the solution isn't feasible we update the cholesky matrix, A and the
        exponentially fading record, v.
        """
        m = len(g)
        feasible = True
        summ = 0

        # Init
        if self.v.shape == (0,):
            self.v = np.zeros((m, self.dim))
        if self.w.shape == (0,):
            self.w = np.ones((m, self.dim))

        for j in range(m):
            if g[j] > 0:
                self.v[j] *= (1 - self.c_c)
                self.v[j] += self.c_c * self.A.dot(self.z)
                self.w[j] = self.invA.dot(self.v[j])
                summ += np.outer(self.v[j], self.w[j]) / self.w[j].T.dot(self.w[j])
                feasible = False

        m_a = sum(g > 0)
        if not feasible:
            self.A -= self.beta / m_a * summ
            if np.isnan(self.A).any():
                print("ERROR: NaN values in the covariance matrix")
                print(f"After {self.count_g} constraint evaluations")
                print(f"summ value: {summ}")
                raise RuntimeError("NaN values in the covariance matrix")

        if m_a > 0:
            ind = g > 0
            U = (- self.v[ind] * self.beta / m_a).T
            V = np.array([self.w[j] / (self.w[j].T @ self.w[j]) for j in np.arange(m)[ind]])
            if m_a == 1:
                C = 1
                self.invA -= (self.invA @ U) * 1/(C + V @ self.invA @ U)[0, 0] * (V @ self.invA)
            else:
                C = np.eye(m_a)
                self.invA -= (self.invA @ U) @ np.linalg.inv(C + V @ self.invA @ U) @ (V @ self.invA)
        self.count_g += 1
        return feasible

    def _updateStepSize(self, lbd):
        """
        Update the value of the step size sigma and the averaged success rate,
        p_succ.
        """
        self.p_succ = (1 - self.c_p) * self.p_succ + self.c_p * lbd
        self.sigma *= np.exp(
            (self.p_succ - self.p_target) / ((1 - self.p_target) * self.d)
        )

    def _updateCholesky(self):
        """
        Update of the cholesky matrix and the exponentially fading record, s,
        in order to change the search space for new candidates.
        Rather than working with the covariance matrix and performing a
        Cholesky decompositionin every iteration of the algorithm, Igel et al.
        presented a direct update of A.
        """
        if self.p_succ < 0.44:
            self.s *= 1 - self.c
            self.s += np.sqrt(self.c * (2 - self.c)) * self.A.dot(self.z)
            self.alpha = 1 - self.c_cov_plus

        else:
            self.s *= 1 - self.c
            self.alpha = 1 - self.c_cov_plus + self.c_cov_plus * self.c * (2 - self.c)

        u = self.invA.dot(self.s)
        u2 = u.T @ u
        self.A *= np.sqrt(self.alpha)
        self.A += np.sqrt(self.alpha) / u2 * (np.sqrt(1 + self.c_cov_plus * u2
                                                      / (self.alpha)) - 1) * np.outer(self.s, u)

        A_temp = self.invA / np.sqrt(self.alpha)
        if u2 != 0:
            A_temp -= (1 - 1 / np.sqrt(1 + self.c_cov_plus * u2/self.alpha)
                       ) / (np.sqrt(self.alpha) * u2) * np.outer(u, u.T.dot(self.invA))
        self.invA = A_temp

        assert not np.isnan(self.A).any()

    def _updateFifthOrder(self):
        """
        In the case where the  solution is worst than the fifth last, we
        incorporate the active covariance matrix update due to Jastrebski and
        Arnold.
        """
        z2 = self.z.T @ self.z
        self.c_cov_minus = np.min(
            [0.4 / (self.dim**(1.6) + 1),
                1 / abs(2 * z2 - 1)])
        a = np.sqrt(1 + self.c_cov_minus)
        b = a / z2 * (np.sqrt(1 - self.c_cov_minus * z2 / (1+self.c_cov_minus)) - 1)
        Az = self.A @ self.z

        self.A *= a
        self.A += b * np.outer(Az, self.z)

        self.invA = self.invA / a - (b / a**2) / (1 + z2 * b / a) * np.outer(self.z, self.z.T.dot(self.invA))
        assert not np.isnan(self.A).any()

    def stop(self, inner=False):
        """
        Stopping criteria
        Set inner to true to test only for criteria related to the inner loop
        """
        if not inner:
            if self.sigma < self.tolsig:
                print("sigma")
                return True
            elif self.stagnation > self.tolstagnation:
                # Stagnation crit
                print("Stagnation crit")
                return True
            elif len(self.best) > 2 and self.best[-2] - self.best[-1] < self.tolfun:
                # TolFun crit
                print("TolFun crit")
                return True
            elif self.sigma * self.p_succ < self.TolX:
                # TolX crit
                print("TolX crit")
                return True
        if self.count_f >= self.tolcountf or self.count_g > self.tolcountg:
            print("Number of evals exceeded")
            return True
        return False


def fmin_con(objective, constraint, x0, sigma0, options=True, plot=False):
    """
    Interface for constrained optimization
    """
    n_f = 0
    n_g = 0
    es = FastActiveElitistES(x0, sigma0)
    sig = []
    vp = []
    std = []
    xs = []
    while not es.stop():
        while True:
            x = es.ask()
            g = constraint(x)
            n_g += 1
            is_feasible = es.test(g)

            # To plot latter
            vp.append(np.linalg.eig(es.A.T.dot(es.A))[0])
            sig.append(es.sigma)
            std.append(np.diag(es.A.dot(es.A)))

            if n_g % 1500 == 0 and options:
                print("{0} evaluation of f and {1} of the constraint."
                      .format(n_f, n_g))

            if is_feasible:
                break
        xs.append(es.x)
        f = objective(x)
        n_f += 1
        es.tell(x, f)
    if plot:
        return es, vp, sig, std, xs
    return es


def fmin2(f, x0, sigma0):
    """
    Standard interface to unconstrained optimization with ActiveElitistES
    """
    es = FastActiveElitistES(x0, sigma0)
    while not es.stop():
        x = es.ask()
        es.tell(x, f(x))

    return es