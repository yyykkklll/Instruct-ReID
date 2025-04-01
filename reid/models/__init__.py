from .pass_transformer_joint import T2IReIDModel

__factory = {
    'T2IReIDModel': T2IReIDModel,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError(f"Unknown model: {name}")
    return __factory[name](**kwargs)
