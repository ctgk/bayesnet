import random
import numpy as np
from bayesnet.network import Network


def hmc(model, call_args, sample_size=100, step_size=0.1, n_step=10):

    if not isinstance(model, Network):
        raise TypeError("model must be Network object")

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
