import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from runner import MyRunner
from elitist_es_fast import fmin_con, fmin, FastActiveElitistES
from problem import LinConsQP
# from scipy.stats import ortho_group
# Import the following package from https://github.com/paulduf/benchmarking_nlco.git
from nlcco.problems import arnold2012

rcParams.update({"font.size": 14})


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
    plt.subplots_adjust(hspace=0.3)

    ax[0, 0].semilogy(sig)
    ax[0, 0].set_title("Evolution of the step size sigma")
    ax[0, 0].grid(True, which="both")
    ax[0, 0].set_xlabel("g-evals")

    diff = np.abs(x - np.array(pb.xopt))
    for i in range(n):
        ax[0, 1].semilogy(diff[:, i], label=i)
    ax[0, 1].set_title("abs(x-x_opt)")
    ax[0, 1].grid(True, which="both")
    ax[0, 1].legend()
    ax[0, 1].set_xlabel("f-evals")

    std = np.array([np.abs(u) for u in stds])  # not sort
    for i in range(n):
        ax[1, 0].plot(np.sqrt(std[:, i])*sig, label=i)
    ax[1, 0].legend()
    ax[1, 0].set_title("Standard deviations times sigma")
    ax[1, 0].set_yscale("log")
    ax[1, 0].grid(True, which="both")
    ax[1, 0].set_xlabel("g-evals")

    vp = np.array([np.sort(np.abs(u)) for u in vps])
    for i in range(n):
        ax[1, 1].semilogy(np.sqrt(vp[:, i]), label=i)
        ax[1, 1].set_title("Evolution of the eigenvalues of the covariance matrix, C")
        ax[1, 1].legend()
    ax[1, 1].grid(True, which="both")
    ax[1, 1].set_xlabel("g-evals")

    name = ["Linear", "Sphere", "Ellipsoïde", "Rotated Ellipsoid"]
    fig.suptitle(f'Problem {name[index]} in dimension {n} with {m} constraints', fontsize=14)
    plt.show()
    return es, vps, sig, stds, x


if __name__ == '__main__':
    # runs(arnold2012)
    es, vps, sig, stds, x = plot_simple(5, 1, 1)
    # pb = LinConsQP(6, 6, 2)
    # print(pb.f(x[-1]), x[-1])
    # for j in range(1, 3):
    #     for i in [5, 10]:
    #         plot_simple(i, int(i/2), j)

# %%
# rcParams.update({"font.size": 14})
n = 5
m = 3
pb = LinConsQP(n, m, 2)
objective = pb.f
constraint = pb.g
x0 = np.ones(n) * n
x_opt = np.array([1]*m + [0]*(n-m))
options = True

n_f = 0
n_g = 0
gvals = []
es = FastActiveElitistES(x0, 1, options)

if False:
    es.c_c = 0
    # input the true gradients of the constraint
    es.v = - np.concatenate((np.eye(m), np.zeros((m, n-m))), axis=1)

sig = []
vps = []
stds = []
xs = [x0]
xg = [x0]
f_vals = [objective(x0)]
feasible_sampled = [0]
# while not es.stop():
while sum(np.abs(xs[-1] - x_opt)) > 1e-5 and n_g < 30000:
    while True:
        # if es.sigma > 10**3:
        #     es.sigma = 1
        x = es.ask()
        g = constraint(x)
        gvals.append(g)
        n_g += 1
        is_feasible = es.test(g)

        # To plot latter
        xg.append(x)
        vps.append(np.linalg.eig(es.A.T.dot(es.A))[0])
        sig.append(es.sigma)
        stds.append(np.diag(es.A.T.dot(es.A)))

        if n_g % 1500 == 0 and options:
            print("{0} evaluation of f and {1} of the constraint."
                  .format(n_f, n_g))

        if is_feasible:
            break

    xs.append(es.x)
    feasible_sampled.append(es.count_g)
    f = objective(x)
    f_vals.append(f)
    n_f += 1
    es.tell(x, f)
print("valeur de f:", f, "et x:", xs[-1])

# Convert data to np arrays
sig = np.asarray(sig).reshape(-1, 1)
stds = np.sqrt(np.asarray(stds))

# Plot
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.3)

is_feasible = np.asarray(gvals) < 0

#diff = np.abs(xs - np.array(pb.xopt))
diff = xg - np.array(pb.xopt)
axis = ax[0, 1]
for i in range(n):
    axis.plot(diff[:, i], label=i)
axis.set_title("$x_i - x^{opt}_i$")
axis.set_yscale("symlog", linthreshy=1e-6)
axis.grid(True, which="both")
axis.legend()
axis.set_xlabel("g-evals")

# #diff = np.abs(xs - np.array(pb.xopt))
# diff = xs - np.array(pb.xopt)
# axis = ax[0, 1]
# for i in range(n):
#     axis.plot(diff[:, i], label=i)
# axis.set_title("$x_i - x^{opt}_i$")
# axis.set_yscale("symlog", linthreshy=1e-6)
# axis.grid(True, which="both")
# axis.legend()
# axis.set_xlabel("f-evals")

axis = ax[0, 0]
axis.semilogy(sig, label="Step size $\sigma$", c="purple")
#axis.set_ylabel("Step size $\sigma$")
secaxis = axis.twinx()
secaxis.semilogy(feasible_sampled, f_vals - pb.f(x_opt), label="$f(x) - f(x^{opt})$", c="black")
#secaxis.set_ylabel("$f(x) - f(x^{opt})$")
#axis.set_title("Evolution of the step size sigma")
#axis.grid(True, which="both")
axis.set_xlabel("g-evals")
axis.legend(title="Left y-axis", loc=(.1,.8), frameon=False)
secaxis.legend(title="Right y- axis", loc=(.65,.1), frameon=False)

#secaxis.legend(title="Right", loc="lower right")


axis = ax[1, 0]

vp = np.array([np.sort(np.abs(u)) for u in vps])
for i in range(n):
    axis.semilogy(np.sqrt(vp[:, i]), c="grey", alpha=.7)
#axis.set_ylabel("Eigenvalues of the covariance matrix C")
#axis.grid(True, which="both")

secaxis = axis.twinx()
metric = stds * sig
for i in range(n):
    secaxis.semilogy(metric[:, i], label=f"i={i}", alpha=.7)
secaxis.legend()
#secaxis.set_ylabel("Standard deviations times sigma: $\sigma C_{i,i}$")
secaxis.set_xlabel("g-evals")

title_left="""Left y-axis
Eigenvalues of C"""
title_right="""Right y-axis: $\sigma C_{i,i}$"""
axis.legend([], [], title=title_left, loc=(.1,.1), frameon=False)
leg = secaxis.legend(title=title_right, loc=(.65,.7), frameon=False, ncol=2)
leg._legend_box.align = "left"


# axis = ax[2, 0]
# gvals = np.asarray(gvals).T
# for i, gval in enumerate(gvals):
#     axis.plot(gval, label=f"$g_{{{i}}}(x)$")
# axis.set_yscale("symlog", linthreshy=1e-6)
# axis.set_title("Constraint functions values")
# axis.grid(True)
# axis.legend()
# axis.set_xlabel("$g$-evals")

# axis = ax[2, 1]
# axis.plot(f_vals - pb.f(x_opt), label="$f(x) - f(x^{opt})$", c="black")
# axis.set_yscale("symlog", linthreshy=1e-6)
# axis.set_title("Objective function")
# axis.grid(True)
# axis.legend()
# axis.set_xlabel("$f$-evals")

axis = ax[1, 1]
diff = xg[1:] - np.array(pb.xopt)
#☺metric = np.abs(diff) * sig / stds
metric = diff / sig / stds

for i in range(n):
    axis.plot(metric[:, i], label=i)
axis.set_title("$(x_i - x^{opt}_i) / \sigma C_{i,i}$")
axis.set_yscale("symlog", linthreshy=1e-2)
axis.grid(True, which="both")
axis.legend(ncol=n)
axis.set_xlabel("g-evals")


name = ["Linear", "Sphere", "Ellipsoïde", "Rotated Ellipsoid"]
fig.suptitle(f'Problem {name[2]} in dimension {n} with {m} constraints',
             fontsize=14)
fig.tight_layout()
plt.show()
