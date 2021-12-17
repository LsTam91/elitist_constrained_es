"""
TODO:
    - check everything (write tests)
    - move TR2 gradients to the nlco library
"""
import numpy as np
from scipy.stats import ortho_group


def arrayize(func):
    def wrapper(x):
        x = np.asarray(x, dtype=float)
        return func(x)
    return wrapper


class LinConsQP:
    """
    Build a constrained continuous optimization problem
    with a convex quadratic objective
    0: Linear
    1: Sphere
    2: Separable ellipsoid
    3: Rotated ellipsoid
    and m axis-aligned, linear constraints x[k] >= 1

    Remarks:
        - If m == 0 then the constraint function (hence also the call method) returns None
        - For the rotated ellipsoid xopt and fmin are not none
    print(problem) for counter infos
    """

    def __init__(self, dimension, nb_constraints=0, problem_index=1):
        self.n = dimension
        self.m = nb_constraints
        assert self.n >= self.m

        self.problem_index = problem_index
        self.objective = problem_infos[self.problem_index]["fun"]

        self.count_f = 0
        if self.m:
            self.count_g = np.zeros(self.m, dtype=int)

        if self.problem_index < 3:
            self.xopt = np.array([1] * self.m + [0] * (self.n - self.m))
            self.f_min = self.objective(self.xopt)
        else:
            self.xopt = None

    def __call__(self, x):
        return self.f(x), self.g(x)

    def f(self, x):
        self.count_f += 1
        return self.objective(x)

    def g(self, x):
        if not self.m:
            return None
        self.count_g += 1
        return np.array([1 - x[k] for k in range(self.m)])

    def gk(self, x, k):
        self.count_g[k] += 1
        return 1 - x[k]

    def __repr__(self):
        problem_data = [
            "%s objective with %s linear constraints" % (
                problem_infos[self.problem_index]["name"], self.m
            )]
        problem_data.append("%s f-evals" % self.count_f)
        if self.m:
            if np.all(self.count_g == self.count_g[0]):
                problem_data.append("%s g-evals" % self.count_g[0])
            else:
                for k in range(self.m):
                    problem_data.append("%s g-%s-evals" % self.count_g[k], k)
        return "\n".join(problem_data)


@arrayize
def linear(x):
    return sum(x)


@arrayize
def sphere(x):
    return sum(x**2)


@arrayize
def elli(x, cond=1e6):
    N = len(x)
    return sum(cond**(np.arange(N) / (N - 1.)) * x**2)


@arrayize
def ellirot(x):
    # TODO: need to fix the seed and generate the Rotation matrix dependant of dimension
    #np.random.seed(seed=1)
    #R = ortho_group.rvs(dimension)
    raise NotImplementedError


problem_infos = {
    0: {"name": "Linear", "fun": linear},
    1: {"name": "Sphere", "fun": sphere},
    2: {"name": "Ellipsoid", "fun": elli},
    3: {"name": "Rotated Ellipsoid", "fun": ellirot}
}