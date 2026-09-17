from .model import SpeechTokenizer
try:
    from .trainer import SpeechTokenizerTrainer
except Exception:
    SpeechTokenizerTrainer = None

__version__ = '1.0.0'
