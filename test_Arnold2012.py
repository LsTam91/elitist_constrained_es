import numpy as np
import matplotlib.pyplot as plt
from runner import MyRunner
from elitist_es import fmin_con, fmin
from problem import LinConsQP
# from scipy.stats import ortho_group
# Import the following package from https://github.com/paulduf/benchmarking_nlco.git
from nlcco.problems import arnold2012


def plot_logger(runner, xopt, name):
    # if end==-1 or end >= len(runner.list_sigma):
    #    end = len(runner.list_sigma)-1

    fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    plt.subplots_adjust(hspace=0.3)

    ax[0, 0].semilogy(runner.list_sigma, label='sigma')
    f = np.array(runner.fct)
    f -= np.min(f)
    f += 10**(-12)
    ax[0, 0].semilogy(f, label='f - min(f)')
    ax[0, 0].legend()
    ax[0, 0].set_title("Evolution of the step size sigma and f - min(f)")
    ax[0, 0].grid(True, which="both")
    ax[0, 0].set_xlabel("g-evals")

    diff = np.abs(runner.list_x - np.array(xopt))
    for i in range(runner.es.dim):
        ax[0, 1].semilogy(diff[:, i], label=i)
    ax[0, 1].set_title("abs(x-x_opt)")
    ax[0, 1].grid(True, which="both")
    ax[0, 1].legend()
    ax[0, 1].set_xlabel("f-evals")

    std = np.array([np.sort(np.abs(u)) for u in runner.std])
    for i in range(runner.es.dim):
        ax[1, 0].plot(np.sqrt(std[:, i])*runner.list_sigma, label=i)
    ax[1, 0].legend()
    ax[1, 0].set_title("Standard deviations times sigma")
    ax[1, 0].grid(True, which="both")
    ax[1, 0].set_xlabel("g-evals")

    vp = np.array([np.sort(np.abs(u)) for u in runner.Q_vp])
    for i in range(runner.es.dim):
        ax[1, 1].semilogy(np.sqrt(vp[:, i]), label=i)
    ax[1, 1].set_title("Evolution of the eigenvalues of the covariance matrix, C")
    ax[1, 1].legend()
    ax[1, 1].grid(True, which="both")
    ax[1, 1].set_xlabel("g-evals")

    fig.suptitle('Problem ' + name, fontsize=14)
    plt.show()


def runs(problems, sigma0=1):
    for pb in problems:

        print("Problem name:", pb)
        problem = arnold2012[pb]["obj"]()
        runner = MyRunner(problem)
        x0 = problem.x_start
        # assert all(problem(x0, add_bounds=True)[1] <= 0)
        runner.run(x0, sigma0)

        plot_logger(runner, problem.xopt, pb)


def plot_simple(n=5, m=3, index=1):
    pb = LinConsQP(n, m, index)
    x0 = np.ones(n)*n
    if m > 0:
        es, vps, sig, stds, x = fmin_con(pb.f, pb.g, x0, 1, plot=True)
    else:
        es, vps, sig, stds, x = fmin(pb.f, x0, 1, plot=True)

    fig, ax = plt.subplots(2, 2, figsize=(16, 8))

    ax[0, 0].semilogy(sig)
    ax[0, 0].set_title("Evolution of the step size sigma")
    ax[0, 0].grid(True, which="both")

    diff = np.abs(x - np.array(pb.xopt))
    for i in range(n):
        ax[0, 1].semilogy(diff[:, i], label=i)
    ax[0, 1].set_title("abs(x-x_opt)")
    ax[0, 1].grid(True, which="both")
    ax[0, 1].legend()
    ax[0, 1].set_xlabel("f-evals")

    std = np.array([np.sort(np.abs(u)) for u in stds])
    for i in range(n):
        ax[1, 0].plot(np.sqrt(std[:, i])*sig, label=i)
    ax[1, 0].legend()
    ax[1, 0].set_title("Standard deviations times sigma")
    ax[1, 0].grid(True, which="both")
    ax[1, 0].set_xlabel("g-evals")

    vp = np.array([np.sort(np.abs(u)) for u in vps])
    for i in range(n):
        ax[1, 1].semilogy(np.sqrt(vp[:, i]), label=i)
        ax[1, 1].set_title("Evolution of the eigenvalues of the covariance matrix, C")
        ax[1, 1].legend()
    ax[1, 1].grid(True, which="both")

    name = ["Linear", "Sphere", "Elli", "Rotated Ellipsoid"]
    fig.suptitle(f'Problem {name[index]} in dimension {n} with {m} constraints', fontsize=14)
    plt.show()


if __name__ == '__main__':
    runs(arnold2012)
    plot_simple(5, 5, 0)

    for j in range(1, 3):
        for i in [5, 10]:
            plot_simple(i, int(i/2), j)
