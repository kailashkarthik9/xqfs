# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from .configuration_mmbt import MMBTConfig
from ...file_utils import is_torch_available

if is_torch_available():
    from .modeling_mmbt import MMBTForClassification, MMBTModel, ModalEmbeddings
