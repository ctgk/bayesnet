import random
import numpy as np
from bayesnet.network import Network


def hmc(model, call_args, parameter=None, sample_size=100, step_size=1e-3, n_step=10):
    """
    Hamiltonian Monte Carlo sampling aka Hybrid Monte Carlo sampling

    Parameters
    ----------
    model : Network
        bayesian network
    call_args : tuple or dict
        observations of the model
    parameter : dict
        dict of parameter to be sampled
    sample_size : int
        number of samples to be generated
    step_size : float
        update size of parameters
    n_step : int
        number of updation of parameters

    Returns
    -------
    sample : dict of list of np.ndarray
        samples from the model given observations
    """

    if not isinstance(model, Network):
        raise TypeError("model must be Network object")

    if not isinstance(sample_size, int):
        raise TypeError(f"sample_size must be int, not {type(sample_size)}")

    if not isinstance(step_size, (int, float)):
        raise TypeError(f"step_size must be float, not {type(step_size)}")

    if not isinstance(n_step, int):
        raise TypeError(f"n_step must be int, not {type(n_step)}")

    def run_model():
        model.clear()
        if isinstance(call_args, tuple):
            model(*call_args)
        elif isinstance(call_args, dict):
            model(**call_args)
        else:
            raise TypeError("call_args must be tuple or dict")

    sample = dict()
    previous = dict()
    velocity = dict()
    if parameter is not None:
        if not isinstance(parameter, dict):
            raise TypeError("parameter must be dict")
        for key, p in parameter.items():
            if p is not model.parameter[key]:
                raise ValueError("parameter must be defined in the model")
        variable = parameter
    else:
        variable = model.parameter

    for key in variable:
        sample[key] = []

    for _ in range(sample_size):
        run_model()
        log_posterior = model.log_pdf()
        log_posterior.backward()
        kinetic_energy = 0
        for key, v in variable.items():
            previous[key] = v.value
            velocity[key] = np.random.normal(size=v.shape)
            kinetic_energy += 0.5 * np.square(velocity[key]).sum()
            velocity[key] += 0.5 * v.grad * step_size
            v.value = v.value + step_size * velocity[key]
        hamiltonian = kinetic_energy - log_posterior.value

        for _ in range(n_step):
            run_model()
            model.log_pdf().backward()
            for key, v in variable.items():
                velocity[key] += step_size * v.grad
                v.value += step_size * velocity[key]

        run_model()
        log_posterior_new = model.log_pdf()
        log_posterior_new.backward()
        kinetic_energy_new = 0
        for key, v in velocity.items():
            v += 0.5 * step_size * variable[key].grad
            kinetic_energy_new += 0.5 * np.square(v).sum()

        hamiltonian_new = kinetic_energy_new - log_posterior_new.value
        accept_proba = np.exp(hamiltonian - hamiltonian_new)

        if random.random() < accept_proba:
            for key, v in variable.items():
                sample[key].append(v.value)
        else:
            for key, v in variable.items():
                v.value = previous[key]
                sample[key].append(v.value)

    return sample
