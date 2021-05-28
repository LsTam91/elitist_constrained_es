import numpy as np


class SphereLinCons:
    """
    Build a constrained continuou optimization problem
    with a sphere objective
    and m axis-aligned, linear constraints x[k] >= 1
    Unconstrained sphere problem if m=0
    print(problem) for counter infos
    """

    def __init__(self, dimension, nb_constraints=0):
        self.n = dimension
        self.m = nb_constraints
        assert self.n >= self.m

        self.count_f = 0
        if self.m:
            self.count_g = np.zeros(self.m, dtype=int)

        self.solution = self.f(np.concatenate([np.ones(self.m), np.zeros(self.n-self.m)]))

    def __call__(self, x):
        return self.f(x), self.g(x)

    def f(self, x):
        self.count_f += 1
        x = np.asarray(x)
        return sum(x**2)

    def g(self, x):
        self.count_g += 1
        return np.array([1 - x[k] for k in range(self.m)])

    def gk(self, x, k):
        self.count_g[k] += 1
        return 1 - x[k]

    def __str__(self):
        problem_data = ["Sphere objective with %s linear constraints" % self.m]
        problem_data.append("%s f-evals" % self.count_f)
        if self.m:
            if np.all(self.count_g == self.count_g[0]):
                problem_data.append("%s g-evals" % self.count_g[0])
            else:
                for k in range(self.m):
                    problem_data.append("%s g-%s-evals" % self.count_g[k], k)
        return "\n".join(problem_data)


class TR2:
    """
    Kramer and Schwefel
    """

    def __init__(self, dimension=2, nb_constraints=1):
        self.n = dimension
        self.m = nb_constraints
        assert self.n >= self.m

        self.count_f = 0
        if self.m:
            self.count_g = np.zeros(self.m, dtype=int)

        self.solution = 2.0

    def __call__(self, x):
        return self.f(x), self.g(x)

    def f(self, x):
        self.count_f += 1
        x = np.asarray(x)
        return sum(x**2)

    def g(self, x):
        self.count_g += 1
        return np.array([2 - x[0] - x[1]])

    def df(self, x):
        return np.array([2*x[0], 2*x[1]])

    def dg(self, x):
        return - np.ones(2)

    def __str__(self):
        problem_data = ["2D Sphere objective with %s linear constraints" % self.m]
        problem_data.append("%s f-evals" % self.count_f)
        if self.m:
            if np.all(self.count_g == self.count_g[0]):
                problem_data.append("%s g-evals" % self.count_g[0])
            else:
                for k in range(self.m):
                    problem_data.append("%s g-%s-evals" % self.count_g[k], k)
        return "\n".join(problem_data)


class elli:

    def __init__(self, dimension, nb_constraints=0):
        self.n = dimension
        self.m = nb_constraints
        assert self.n >= self.m

        self.count_f = 0
        if self.m:
            self.count_g = np.zeros(self.m, dtype=int)

        self.solution = 2.0

    def __call__(self, x):
        return self.f(x), self.g(x)

    def f(self, x, cond=1e6):
        self.count_f += 1
        N = len(x)
        return sum(cond**(np.arange(N) / (N - 1.)) * x**2)

    def g(self, x):
        self.count_g += 1
        return np.array([1 - x[k] for k in range(self.m)])

    def __str__(self):
        problem_data = ["elli objective with %s linear constraints" % self.m]
        problem_data.append("%s f-evals" % self.count_f)
        if self.m:
            if np.all(self.count_g == self.count_g[0]):
                problem_data.append("%s g-evals" % self.count_g[0])
            else:
                for k in range(self.m):
                    problem_data.append("%s g-%s-evals" % self.count_g[k], k)
        return "\n".join(problem_data)
