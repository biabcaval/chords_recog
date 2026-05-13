import yaml


# TODO: add function should be changed
class HParams(object):
    # Hyperparameter class using yaml
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def add(self, **kwargs):
        # change is needed - if key is existed, do not update.
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    def get(self, key, default=None):
        # Mimic ``dict.get`` for top-level sections so callers like
        # ``config.get('gradnorm', {})`` actually return the YAML block
        # instead of silently falling back to the default.
        return self.__dict__.get(key, default)

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)
        return self

    def __repr__(self):
        return '\nHyperparameters:\n' + '\n'.join([' {}={}'.format(k, v) for k, v in self.__dict__.items()])

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            return cls(**yaml.safe_load(f))


if __name__ == '__main__':
    hparams = HParams.load('hparams.yaml')
    print(hparams)
    d = {"MemoryNetwork": 0, "c": 1}
    hparams.add(**d)
    print(hparams)
