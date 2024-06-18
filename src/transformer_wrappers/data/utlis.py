from .open_assistant_guanaco import OpenAssistantGuanaco
from .book_corpus import BookCorpus
from .wiki_text import WikiText2

from torch.utils.data import Dataset
from typing import Dict, Type


corpus_mapping: Dict[str, Type[Dataset]] = {
    OpenAssistantGuanaco.__name__: OpenAssistantGuanaco,
    BookCorpus.__name__: BookCorpus,
    WikiText2.__name__: WikiText2
}
