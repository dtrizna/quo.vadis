from collections.abc import Iterable
from numpy import exp


def sigmoid(X):
   return 1/(1+exp(-X))


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
