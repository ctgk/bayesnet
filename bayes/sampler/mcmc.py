import random
import numpy as np
from bayes.random.random import RandomVariable
from bayes.random.gaussian import Gaussian


def all_variable_in(model):
    rv = {}
    for param in model.parameter.values():
        if isinstance(param, RandomVariable):
            if param.name is None:
                raise ValueError("name of RandomVariable must be specified")
            rv[param.name] = param
            rv.update(all_variable_in(param))
    return rv


def mcmc(model, sample_size=100, downsample=1, **kwargs):
    sample = dict()
    previous = dict()
    proposal = kwargs
    variable = all_variable_in(model)

    for key in variable:
        sample[key] = []
        if key not in proposal:
            proposal[key] = Gaussian(0, 1)
        else:
            if not isinstance(proposal[key], RandomVariable):
                raise TypeError("proposal distribution must be RandomVariable")

    for rv in variable.values():
        rv.observe(rv.draw())

    log_pdf = 0
    for rv in variable.values():
        log_pdf += np.sum(rv.log_pdf())
    log_pdf += np.sum(model.log_pdf())

    for _ in range(sample_size // 10):
        for key, rv in variable.items():
            previous[key] = rv.data
            rv.observe(rv.data + proposal[key].draw())

        log_pdf_new = 0
        for key, rv in variable.items():
            log_pdf_new += np.sum(rv.log_pdf())
        log_pdf_new += np.sum(model.log_pdf())
        accept_proba = np.exp(log_pdf_new - log_pdf)

        if random.random() < accept_proba:
            pass
        else:
            for key, rv in variable.items():
                rv.observe(previous[key])

    for i in range(1, sample_size * downsample + 1):

        for key, rv in variable.items():
            previous[key] = rv.data
            rv.observe(rv.data + proposal[key].draw())

        log_pdf_new = 0
        for key, rv in variable.items():
            log_pdf_new += np.sum(rv.log_pdf())
        log_pdf_new += np.sum(model.log_pdf())
        accept_proba = np.exp(log_pdf_new - log_pdf)

        if i % downsample != 0:
            if random.random() > accept_proba:
                for key, rv in variable.items():
                    rv.observe(previous[key])
        else:
            if random.random() < accept_proba:
                for key, rv in variable.items():
                    sample[key].append(rv.data)
                    log_pdf = log_pdf_new
            else:
                for key, rv in variable.items():
                    sample[key].append(previous[key])
                    rv.observe(previous[key])

    for rv in variable.values():
        rv.data = None

    return sample
