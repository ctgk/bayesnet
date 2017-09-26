from bayesnet.tensor.parameter import Parameter


class Network(object):

    def __init__(self, **kwargs):
        self.parameter = {}
        for key, value in kwargs.items():
            if isinstance(value, Parameter):
                self.parameter[key] = value
            else:
                try:
                    value = Parameter(value)
                except TypeError:
                    raise TypeError(f"invalid type argument: {type(value)}")
                self.parameter[key] = value
            object.__setattr__(self, key, value)

    def cleargrad(self):
        for p in self.parameter.values():
            p.cleargrad()
