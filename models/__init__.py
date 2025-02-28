from ._base import get_model, register_model
from .fno import FNO, FNOVanilla
from .functional import FFM, ConditionalFFM


def get_flow_model(model_cfg, encoder_cfg, conditional=False):
    """
    Build the functional flow model.
    :param model_cfg: model configs passed to the flow model, type indicates the model type
    :param encoder_cfg: encoder configs passed to the encoder model
    :param conditional: whether the model is conditional
    :return: the flow model
    """
    model_factory = FFM if not conditional else ConditionalFFM
    return model_factory(get_model(encoder_cfg), **model_cfg)
