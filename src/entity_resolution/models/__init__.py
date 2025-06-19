# Import modules conditionally to handle missing dependencies
try:
    from .retriever import EntityRetriever
except ImportError:
    EntityRetriever = None

try:
    from .reader import EntityReader
except ImportError:
    EntityReader = None

try:
    from .consensus import ConsensusModule
except ImportError:
    ConsensusModule = None

try:
    from .output import EntityOutputFormatter
except ImportError:
    EntityOutputFormatter = None

try:
    from .atg import ATGModel
except ImportError:
    ATGModel = None
