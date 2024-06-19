from .book_corpus import BookCorpus
from .open_assistant_guanaco import OpenAssistantGuanaco
from .project_gutemberg import PG19
from .web_text import OpenWebText
from .wiki_text import WikiText2, WikiText103

from torch.utils.data import Dataset
from typing import Dict, Type


corpus_mapping: Dict[str, Type[Dataset]] = {
    BookCorpus.__name__: BookCorpus,
    OpenAssistantGuanaco.__name__: OpenAssistantGuanaco,
    OpenWebText.__name__: OpenWebText,
    PG19.__name__: PG19,
    WikiText2.__name__: WikiText2,
    WikiText103.__name__: WikiText103
}
