# -*- coding: utf-8 -*-
import numpy as np
from elitist_es import ActiveElitistES, FastActiveElitistES
from codetiming import Timer
from problem import LinConsQP
from pandas import read_csv
import matplotlib.pyplot as plt


def run_es(ActiveElitistES, objective, constraint, dim, m):
    x0 = np.random.randint(1, 10, dim) * dim * 5
    f = objective(x0)
    n_f = 0
    n_g = 0
    es = ActiveElitistES(x0, 1)
    x_opt = [1] * m + [0] * (dim - m)
    f_opt = objective(x_opt)

    while not np.abs(f_opt - f) < 1e-5:
        while True:
            x = es.ask()
            g = constraint(x)
            n_g += 1
            is_feasible = es.test(g)
            if is_feasible:
                break
        f = objective(x)
        n_f += 1
        es.tell(x, f)
    return f"{np.abs(f_opt - f)} in {n_f} f evaluation and {n_g} g evaluation"


def time_diff(objective, constraint, dim, m,
              fast=FastActiveElitistES, slow=ActiveElitistES, niter=20):
    data = []
    for i in range(niter):
        Timer.timers.clear()
        timer_options = {"logger": None}
        timer_slow = Timer("slow", **timer_options)
        timer_fast = Timer("fast", **timer_options)

        with timer_slow:
            print(run_es(slow, objective, constraint, dim, m))

        with timer_fast:
            print(run_es(fast, objective, constraint, dim, m))

        datum = (dim, Timer.timers['slow'], Timer.timers['fast'])
        data.append(datum)
        print('iter:', i, 'time', datum)
    return data


if __name__ == '__main__':
    quick = []
    niter = 10
    for dim in [5, 15, 30, 50, 100, 200]:  # become really long for dim>200
        m = 1
        pb = LinConsQP(dim, m, 1)
        data = time_diff(pb.f, pb.g, dim, m, niter=niter)
        quick.append(np.sum(data, axis=0)/niter)
        data_str = [" ".join(list(map(str, datum))) for datum in data]
        with open(f"timer_m={m}_{niter}runs", "a") as logfile:
            logfile.write("\n".join(data_str))
            logfile.write("\n")


# %%
data = read_csv(f"data/timer_m=1_10runs",
                sep=" ",
                header=None,
                names=["dim", "slow", "fast"])
nb = int(len(data)/niter)
ind = [i*niter for i in range(nb)]
dimension = data["dim"][ind]
normal_es = [sum(data["slow"][i*niter: (i+1)*niter])/niter for i in range(nb)]
fast_es = [sum(data["fast"][i*niter: (i+1)*niter])/niter for i in range(nb)]

plot_options = {"linewidth": 1.5, "marker": "x", "markersize": 4}
fig, ax = plt.subplots()
ax.plot(dimension, normal_es, **plot_options, color='red', label="classical")
ax.plot(dimension, fast_es, **plot_options, color='green', label="faster")

plt.grid(True, which="both", alpha=.7)
plt.legend(ncol=2, loc="upper left")
plt.title("Time difference between classical implementation and our")
plt.show()