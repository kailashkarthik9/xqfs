# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from .tokenization_transfo_xl import TransfoXLCorpus, TransfoXLTokenizer
from ...file_utils import is_tf_available, is_torch_available

if is_torch_available():
    from .modeling_transfo_xl import (
        TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
        AdaptiveEmbedding,
        TransfoXLLMHeadModel,
        TransfoXLModel,
        TransfoXLPreTrainedModel,
        load_tf_weights_in_transfo_xl,
    )

if is_tf_available():
    from .modeling_tf_transfo_xl import (
        TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFAdaptiveEmbedding,
        TFTransfoXLLMHeadModel,
        TFTransfoXLMainLayer,
        TFTransfoXLModel,
        TFTransfoXLPreTrainedModel,
    )
