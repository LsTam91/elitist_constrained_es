#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launcher for any number of batches.

Example to launch 8 batches::

    python launch.py example_experiment2.py budget_multiplier=3 8

executes::

    nohup nice python -u example_experiment2.py budget_multiplier=3 batch=i/8

for ``i in range(8)``.

After that::

    ps ax -o  uid,uname,pid,ni,%mem,%cpu,time,etime,pid,cmd | sort -n  | tail -n 33

shows running processes.

"""
import os


def cmd(s):
    print(s)
    os.system(s)  # comment for dry run


if __name__ == "__main__":

    python_name = "python3"  # './python' +     os.getcwd()[-3:]
    script_name = f"run_elces.py"

    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

    batches = 5
    algorithms = [3]#, 2, 3, 4]
    budget = 1.6e3

    for i in range(batches):
        suffix = f"elces_batch{i:03d}"
        cmd((f"nohup nice {python_name} -u {script_name} "
            + f"budget_multiplier={budget} "
            + f"batch={i}/{batches} "
            + f"> logs/out_{suffix}.txt 2> logs/err_{suffix}.txt &"))


