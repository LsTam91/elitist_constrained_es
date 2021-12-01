# Python implementation for the (1+1)-CMA ES

## In this repository:

- A Computational Efficient (1+1)-CMA-ES
    - ask-and-tell interface via `CholeskyElitistES`
    - direct interface via `fmin`
- A (1+1)-CMA-ES for constrained optimisation (_in progress_)
    - ask-and-tell interface via `ActiveElitistES`
    - direct interface via `fmin_con`
- Test problems with display utilities

Everything implemented with Python and NumPy.

## References:

1. Igel, C., Suttorp, T. & Hansen, N.
_A Computational Efficient Covariance Matrix Update and a (1+1)-CMA for Evolution Strategies._
in Proceedings of the eighth international conference on Genetic and evolutionary computation conference - GECCO ’06 (ACM Press, 2006)

2. Arnold, D. V. & Hansen, N.
_A (1+1)-CMA-ES for constrained optimisation._
in Proceedings of the fourteenth international conference on Genetic and evolutionary computation conference - GECCO ’12 297 (ACM Press, 2012). doi:10.1145/2330163.2330207.



