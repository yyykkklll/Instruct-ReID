from .pass_transformer_joint import PASS_Transformer_DualAttn_joint

__factory = {
    'PASS_Transformer_DualAttn_joint': PASS_Transformer_DualAttn_joint,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError(f"Unknown model: {name}")
    return __factory[name](**kwargs)