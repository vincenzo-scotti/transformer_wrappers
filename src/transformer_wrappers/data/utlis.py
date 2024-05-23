from .open_assistant_guanaco import OpenAssistantGuanaco

from torch.utils.data import Dataset
from typing import Dict, Type


corpus_mapping: Dict[str, Type[Dataset]] = {
    OpenAssistantGuanaco.__name__: OpenAssistantGuanaco
}
