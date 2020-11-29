# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from .configuration_squeezebert import SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, SqueezeBertConfig
from .tokenization_squeezebert import SqueezeBertTokenizer
from ...file_utils import is_tokenizers_available, is_torch_available

if is_tokenizers_available():
    from .tokenization_squeezebert_fast import SqueezeBertTokenizerFast

if is_torch_available():
    from .modeling_squeezebert import (
        SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        SqueezeBertForMaskedLM,
        SqueezeBertForMultipleChoice,
        SqueezeBertForQuestionAnswering,
        SqueezeBertForSequenceClassification,
        SqueezeBertForTokenClassification,
        SqueezeBertModel,
        SqueezeBertModule,
        SqueezeBertPreTrainedModel,
    )
