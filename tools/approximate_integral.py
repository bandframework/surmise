import math


def approximate_integral(f, samples):
    # According to the docs, this routine can be a bit slower but tries to
    # minimize numerical error better than other summation functions.
    return math.fsum(f(samples)) / float(len(samples))
