import random
import numpy as np
from bayesnet.network import Network


def hmc(model, call_args, sample_size=100, step_size=0.1, n_step=10):

    if not isinstance(model, Network):
        raise TypeError("model must be Network object")

    def run_model():
        if isinstance(call_args, tuple):
            model(*call_args)
        elif isinstance(call_args, dict):
            model(**call_args)
        else:
            raise TypeError("call_args must be tuple or dict")

    def leapfrog(model, velocity):
        run_model()
        model.log_pdf().backward()
        for key, v in variable.items():
            velocity[key] += 0.5 * v.grad * step_size
            v.value = v.value + step_size * velocity[key]

        for _ in range(n_step):
            run_model()
            model.log_pdf().backward()
            for key, v in variable.items():
                velocity[key] += step_size * v.grad
                v.value += step_size * velocity[key]

        run_model()
        model.log_pdf().backward()
        for key, v in velocity.items():
            v += 0.5 * step_size * variable[key].grad

    sample = dict()
    previous = dict()
    velocity = dict()
    velocity_prev = dict()
    variable = model.parameter

    for key in variable:
        sample[key] = []

    for _ in range(sample_size):
        run_model()
        log_posterior = model.log_pdf().value
        kinetic_energy = 0
        for key, v in variable.items():
            previous[key] = v.value
            velocity[key] = np.random.normal(size=v.shape)
            velocity_prev[key] = velocity[key] * 1
            kinetic_energy += 0.5 * np.square(velocity[key]).sum()
        hamiltonian = log_posterior + kinetic_energy

        leapfrog(model, velocity)

        run_model()
        log_posterior_new = model.log_pdf().value
        kinetic_energy_new = 0
        for v in velocity.values():
            kinetic_energy_new += 0.5 * np.square(v).sum()
        hamiltonian_new = log_posterior_new + kinetic_energy_new
        accept_proba = np.exp(hamiltonian_new - hamiltonian)
        if not 0.9 < accept_proba < 1.1:
            print(accept_proba, log_posterior, kinetic_energy, log_posterior_new, kinetic_energy_new)
            print(velocity_prev["w"], previous["w"],velocity["w"], variable["w"].value)
        if random.random() < accept_proba:
            for key, v in variable.items():
                sample[key].append(v.value)
        else:
            for key, v in variable.items():
                v.value = previous[key]
                sample[key].append(v.value)

    return sample
