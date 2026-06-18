"""
Model definitions for chord recognition.
"""
from models.btc_model import BTC_model
from models.baseline_models import CNN, CRNN, Crf
from models.crf_model import CRF
from models.beats_chord_model import (
    BEATsChordDecomposer,
    LinearClassifier,
    MLPClassifier,
)

__all__ = [
    'BTC_model', 'CNN', 'CRNN', 'Crf', 'CRF',
    'BEATsChordDecomposer', 'LinearClassifier', 'MLPClassifier',
]

