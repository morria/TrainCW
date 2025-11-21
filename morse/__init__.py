"""
Morse code utilities for encoding and timing calculations.
"""

from .morse_code import MORSE_CODE_DICT, MorseCode
from .timing import WPM_TO_UNIT_TIME, TimingCalculator


__all__ = ["MORSE_CODE_DICT", "WPM_TO_UNIT_TIME", "MorseCode", "TimingCalculator"]
