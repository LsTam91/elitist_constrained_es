import numpy as np
import matplotlib.pyplot as plt
from elitist_es import MyRunner, ActiveElitistES, fmin
from problem import SphereLinCons, elli
# from scipy.stats import ortho_group
# Import the following package from https://github.com/paulduf/benchmarking_nlco.git
from nlcco.base import BaseRunner
from nlcco.problems import arnold2012, LinConsQP


def plot_logger(runner, xopt, name):
    # if end==-1 or end >= len(runner.list_sigma):
    #    end = len(runner.list_sigma)-1

    fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    plt.subplots_adjust(hspace=0.3)

    ax[0, 0].semilogy(runner.list_sigma)
    # ax[0, 0].semilogy(runner.A_norm)
    ax[0, 0].set_title("Evolution of the step size sigma")
    ax[0, 0].grid(True, which="both")

    diff = np.abs(runner.list_x - np.array(xopt))
    for i in range(runner.es.dim):
        ax[0, 1].semilogy(diff[:, i], label=i)
        ax[0, 1].set_title("abs(x-x_opt)")
        ax[0, 1].grid(True, which="both")
        ax[0, 1].legend()

    f = np.array(runner.es.fct)
    f -= np.min(f)
    f += 10**(-12)
    ax[1, 0].semilogy(f)
    ax[1, 0].set_title("f - min(f)")
    ax[1, 0].grid(True, which="both")

    vp = np.array([np.sort(np.abs(u)) for u in runner.Q_vp])
    for i in range(runner.es.dim):
        ax[1, 1].semilogy(np.sqrt(vp[:, i]), label=i)
        ax[1, 1].set_title("Evolution of the eigenvalues of the covariance matrix, C")
        ax[1, 1].legend()
    ax[1, 1].grid(True, which="both")

    fig.suptitle('Problem ' + name, fontsize=14)
    plt.show()


def runs(problems, sigma0=1):
    for pb in problems:
        if pb == 'TR2':
            break
        print("Problem name:", pb)
        problem = arnold2012[pb]["obj"]()
        runner = MyRunner(problem)
        x0 = problem.x_start
        # assert all(problem(x0, add_bounds=True)[1] <= 0)
        runner.run(x0, sigma0)

        plot_logger(runner, problem.xopt, pb)


def plot_simple(vp, sig):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].semilogy(sig)
    ax[0].set_title("Evolution of the step size sigma")
    ax[0].grid(True, which="both")

    vp = np.array([np.sort(np.abs(u)) for u in vp])
    for i in range(len(vp[0])):
        ax[1].semilogy(np.sqrt(vp[:, i]), label=i)
        ax[1].set_title("Evolution of the eigenvalues of the covariance matrix, C")
        ax[1].legend()
    ax[1].grid(True, which="both")

    plt.show()


if __name__ == '__main__':
    runs(arnold2012)

# %%

dimension = 5
x0 = np.ones(dimension) * dimension
sigma0 = 1

problem = elli(dimension, 0)
es, vp, sigmas = fmin(problem.f, x0, sigma0, options=True)
plot_simple(vp, sigmas)
print(problem)
print(es.x)
