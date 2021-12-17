if __name__ == "__main__":
    import numpy as np
    from elces import fmin2, fmin_con
    from elces.problem import LinConsQP

    dimension = 5
    m = int(dimension / 2)
    x0 = np.ones(dimension) * dimension
    sigma0 = 1

    problem = LinConsQP(dimension, 0, 1)
    es = fmin2(problem.f, x0, sigma0)
    print(problem)
    print(es.x)

    problem = LinConsQP(dimension, m, 1)
    es = fmin_con(problem.f, problem.g, x0, sigma0)
    print(problem)
    print(es.x)
