es = None

while not es.stop():
    x = es.ask()
    g = constraint(x)
    if all(g <= 0):
        f = objective(x)
        es.tell(x, f)
    else:
        es.update_from_infeasible(x)

"""
Remarks:
- es.update_from_infeasible only updates the CM.
    You could think of an algorithm which updates the mean (not a good idea as we would like to move the mean away, not closer to the bad point)
- Discussion of extension of this for the comma setting
- Sort feasible and infeasible chacun de leur coté, assigner aux solutions infeasible
    la sum des g^2 pour le ranking
- main difference between this and death penalty is that you don't update the step size when you have a failure
- Do a timeline issue on Github
"""

"""
Meeting Louis:

- Remarks code
    - logging just like in pycma
        - check cma.logger.Logger and DummyLogger
    - Ptarget is not parametrized
    - Because the number of f and g values are not the same, we may want to relate them on the same scale

- Suggestion to fix the divergence/convergence of stepsize/CM issue:
    - normalize the trace of the CM back to n
    - Check the math to assess that the active update for the Cholesky decomposition the trace is decreasing and that the normalization is the way to go to fix it
"""