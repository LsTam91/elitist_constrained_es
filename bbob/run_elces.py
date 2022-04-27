#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Running experiments with AL-CMA-ES

From the script: A short and simple example experiment with restarts.

The script is fully functional but also emphasises on readability. It
features restarts, timings and recording termination conditions.

When calling the script, previously assigned variables can be re-assigned
via a ``name=value`` argument without white spaces, where ``value`` is
interpreted as a single python literal. Additionally, ``batch`` is recognized
as argument defining the `current_batch` number and the number of `batches`,
like ``batch=2/8`` runs batch 2 of 8.

Run it like this

    example_experiment2.py budget_multiplier=1000 batch=1/16 algorithm_version=1

Post-processing is removed as this should to be used in multiple-batches setting

Details: ``batch=9/8`` is equivalent to ``batch=1/8``. The first number
is taken modulo to the second.

See the code: `<https://github.com/numbbo/coco/blob/master/code-experiments/build/python/example_experiment2.py>`_
"""
from __future__ import division, print_function, unicode_literals
__author__ = "Nikolaus Hansen and Paul Dufossé and Asma Atamna"
import os, sys
import time  # output some timings per evaluation
from collections import defaultdict
import numpy as np  # for median, zeros, random, asarray
import cocoex  # experimentation module
import argparse


### MKL bug fix
def set_num_threads(nt=1, disp=1):
    """see https://github.com/numbbo/coco/issues/1919
    and https://twitter.com/jeremyphoward/status/1185044752753815552
    """
    try: import mkl
    except ImportError: disp and print("mkl is not installed")
    else:
        mkl.set_num_threads(nt)
    nt = str(nt)
    for name in ['OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS',
                 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS']:
        os.environ[name] = nt
    disp and print("setting mkl threads num to", nt)

if sys.platform.lower() not in ('darwin', 'windows'):
    set_num_threads(1)

### solver imports (add other imports if necessary)
import elces

def fmin_con_alcmaes(problem, x0, maxevals):
    """Minimization of a problem from the bbob-constrained test suite using (1+1)-CMA-ES

    From the elces package. Problem is a callable such that problem(x) is the objective
    and problem.constraint(x) returns a vector-valued constraint function
    with convention g(x) <= 0

    Parameters:
        - problem: callable as described
        - x0: array-like of size problem.dimension
        - maxevals : integer >= 0

    Returns:
        - ES object
    """

    sigma0 = 2  # standard choice would be 2 here
    options = {"tolcount": {"fg": maxevals}}
    try:
        es = elces.fmin_con(
            problem, problem.constraint, x0, sigma0, disp=False, options=options
        )
        return es.stop()
    except RuntimeError:
        return "Runtime error"

suite_name = "bbob-constrained"  # see cocoex.known_suite_names
budget_multiplier = 100  # times dimension, increase to 10, 100, ...
suite_filter_options = (""  # without filtering a suite has instance_indices 1-15
                        #"dimensions: 2"  # skip dimension 40
                        # "instance_indices: 1-5 "  # relative to suite instances
                        # "year:2019 "  # select instances by year
                       )
# for more suite filter options see http://numbbo.github.io/coco-doc/C/#suite-parameters
batches = 1  # number of batches, batch=3/32 works to set both, current_batch and batches
current_batch = 1  # only current_batch modulo batches is relevant
output_folder = ''

### possibly modify/overwrite above input parameters from input args
if __name__ == "__main__":


    if len(sys.argv) > 1 and sys.argv[1] in ('-h', 'help', '-help', '--help'):
        print(__doc__)
        raise ValueError("printed help and aborted")
    input_params = cocoex.utilities.args_to_dict(
        sys.argv[1:], globals(),
        {'batch': 'current_batch/batches'},
        print=print)
    globals().update(input_params)  # (re-)assign variables

# extend output folder input parameter, comment out if desired otherwise
output_folder += '%s_%dD_on_%s' % (
        elces.__name__, int(budget_multiplier), suite_name)

if batches > 1:
    output_folder += "_batch%03dof%d" % (current_batch, batches)

### prepare
suite = cocoex.Suite(suite_name, "", suite_filter_options)
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()
stoppings = defaultdict(list)  # dict of lists, key is the problem index
timings = defaultdict(list)  # key is the dimension

### go
print('*** benchmarking %s on suite %s ***'
      % (elces.__name__, suite_name))
time0 = time.time()
for batch_counter, problem in enumerate(suite):  # this loop may take hours or days...
    if batch_counter % batches != current_batch % batches:
        continue
    if not len(timings[problem.dimension]) and len(timings) > 1:
        print("\n   %s %d-D done in %.1e seconds/evaluations"
              % (minimal_print.stime, sorted(timings)[-2],
                 np.median(timings[sorted(timings)[-2]])), end='')
    problem.observe_with(observer)  # generate the data for cocopp post-processing
    problem(np.zeros(problem.dimension))  # making algorithms more comparable
    propose_x0 = problem.initial_solution_proposal  # callable, all zeros in first call
    evalsleft = lambda: int(problem.dimension * budget_multiplier + 1 -
                            (problem.evaluations + problem.evaluations_constraints))
    time1 = time.time()
    # apply restarts
    irestart = -1
    while evalsleft() > 0 and not problem.final_target_hit:
        irestart += 1

        stop = fmin_con_alcmaes(problem, propose_x0(), maxevals=evalsleft())
        stoppings[problem.index].append((problem.evaluations, stop))

    timings[problem.dimension].append((time.time() - time1) / problem.evaluations
                                      if problem.evaluations else 0)
    minimal_print(problem, restarted=irestart, final=problem.index == len(suite) - 1)
    with open(output_folder + '_stopping_conditions.pydict', 'wt') as file_:
        file_.write("# code to read in these data:\n"
                    "# import ast\n"
                    "# with open('%s_stopping_conditions.pydict', 'rt') as file_:\n"
                    "#     stoppings = ast.literal_eval(file_.read())\n"
                    % output_folder)
        file_.write(repr(dict(stoppings)))

### print timings and final message
print("\n   %s %d-D done in %.1e seconds/evaluations"
      % (minimal_print.stime, sorted(timings)[-1], np.median(timings[sorted(timings)[-1]])))
if batches > 1:
    print("*** Batch %d of %d batches finished in %s."
          " Make sure to run *all* batches (via current_batch or batch=#/#) ***"
          % (current_batch, batches, cocoex.utilities.ascetime(time.time() - time0)))
else:
    print("*** Full experiment done in %s ***"
          % cocoex.utilities.ascetime(time.time() - time0))

print("Timing summary:\n"
      "  dimension  median seconds/evaluations\n"
      "  -------------------------------------")
for dimension in sorted(timings):
    print("    %3d       %.1e" % (dimension, np.median(timings[dimension])))
print("  -------------------------------------")
