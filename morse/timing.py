"""
Morse code timing calculations and WPM conversions.
"""

import numpy as np
from typing import Dict, Optional


def WPM_TO_UNIT_TIME(wpm: float) -> float:
    """
    Convert WPM (Words Per Minute) to unit time in seconds.

    Uses the PARIS standard: The word "PARIS" has 50 units.
    At N WPM, "PARIS" is sent N times per minute.

    Args:
        wpm: Words per minute

    Returns:
        Duration of one unit (dit) in seconds
    """
    return 1.2 / wpm  # 60 seconds / (50 units * wpm)


class TimingCalculator:
    """
    Calculate morse code timing with support for variance and imperfections.
    """

    def __init__(self, wpm: float, timing_variance: Optional[Dict[str, float]] = None):
        """
        Initialize timing calculator.

        Args:
            wpm: Words per minute (5-40 typical range)
            timing_variance: Optional dict with variance parameters:
                - 'dit_dah_ratio': Ratio of dah to dit (default 3.0, vary 2.3-3.7)
                - 'element_gap_variance': Variance in element gaps (0-0.35)
                - 'char_gap_variance': Variance in character gaps (0-0.30)
                - 'word_gap_variance': Variance in word gaps (0-0.25)
        """
        self.wpm = wpm
        self.unit_time = WPM_TO_UNIT_TIME(wpm)

        # Default timing parameters
        self.dit_dah_ratio = 3.0
        self.element_gap_variance = 0.0
        self.char_gap_variance = 0.0
        self.word_gap_variance = 0.0

        # Apply variance if provided
        if timing_variance:
            self.dit_dah_ratio = timing_variance.get('dit_dah_ratio', 3.0)
            self.element_gap_variance = timing_variance.get('element_gap_variance', 0.0)
            self.char_gap_variance = timing_variance.get('char_gap_variance', 0.0)
            self.word_gap_variance = timing_variance.get('word_gap_variance', 0.0)

    def get_dit_duration(self) -> float:
        """Get dit duration in seconds."""
        return self.unit_time

    def get_dah_duration(self) -> float:
        """Get dah duration in seconds."""
        return self.unit_time * self.dit_dah_ratio

    def get_element_gap_duration(self) -> float:
        """Get inter-element gap duration in seconds (with variance)."""
        base_duration = self.unit_time
        variance = np.random.uniform(-self.element_gap_variance, self.element_gap_variance)
        return max(0.001, base_duration * (1.0 + variance))

    def get_char_gap_duration(self) -> float:
        """Get inter-character gap duration in seconds (with variance)."""
        base_duration = self.unit_time * 3
        variance = np.random.uniform(-self.char_gap_variance, self.char_gap_variance)
        return max(0.001, base_duration * (1.0 + variance))

    def get_word_gap_duration(self) -> float:
        """Get inter-word gap duration in seconds (with variance)."""
        base_duration = self.unit_time * 7
        variance = np.random.uniform(-self.word_gap_variance, self.word_gap_variance)
        return max(0.001, base_duration * (1.0 + variance))

    def get_timing_sequence(self, elements: list) -> list:
        """
        Convert element list to timing sequence.

        Args:
            elements: List of (element_type, character) tuples from MorseCode.text_to_elements()

        Returns:
            List of (duration_seconds, is_tone_on) tuples
        """
        timing_sequence = []

        for element_type, _ in elements:
            if element_type == 'dit':
                timing_sequence.append((self.get_dit_duration(), True))
            elif element_type == 'dah':
                timing_sequence.append((self.get_dah_duration(), True))
            elif element_type == 'element_gap':
                timing_sequence.append((self.get_element_gap_duration(), False))
            elif element_type == 'char_gap':
                timing_sequence.append((self.get_char_gap_duration(), False))
            elif element_type == 'word_gap':
                timing_sequence.append((self.get_word_gap_duration(), False))

        return timing_sequence


def select_operator_style() -> Dict[str, float]:
    """
    Randomly select an operator sending style with realistic timing variance.

    Returns:
        Dict with timing variance parameters
    """
    style = np.random.choice(['clean', 'typical', 'rushed', 'stretched'],
                            p=[0.50, 0.30, 0.15, 0.05])

    if style == 'clean':
        # Clean sender (±10% variance)
        return {
            'dit_dah_ratio': np.random.normal(3.0, 0.1),
            'element_gap_variance': np.random.uniform(0.0, 0.10),
            'char_gap_variance': np.random.uniform(0.0, 0.10),
            'word_gap_variance': np.random.uniform(0.0, 0.10),
        }
    elif style == 'typical':
        # Typical operator (±20% variance)
        return {
            'dit_dah_ratio': np.random.normal(3.0, 0.2),
            'element_gap_variance': np.random.uniform(0.10, 0.20),
            'char_gap_variance': np.random.uniform(0.10, 0.20),
            'word_gap_variance': np.random.uniform(0.05, 0.15),
        }
    elif style == 'rushed':
        # Rushed/sloppy (±30% variance, compressed gaps)
        return {
            'dit_dah_ratio': np.random.normal(2.8, 0.3),  # Shorter dahs
            'element_gap_variance': np.random.uniform(0.20, 0.35),
            'char_gap_variance': np.random.uniform(-0.30, 0.10),  # Compressed
            'word_gap_variance': np.random.uniform(-0.25, 0.10),  # Compressed
        }
    else:  # stretched
        # Stretched style (longer dahs, wider spacing)
        return {
            'dit_dah_ratio': np.random.normal(3.3, 0.2),  # Longer dahs
            'element_gap_variance': np.random.uniform(0.05, 0.15),
            'char_gap_variance': np.random.uniform(0.15, 0.30),  # Wider gaps
            'word_gap_variance': np.random.uniform(0.10, 0.25),
        }


def sample_wpm(phase: int = 3) -> float:
    """
    Sample WPM from distribution based on curriculum phase.

    Args:
        phase: Curriculum phase (1=foundation, 2=expansion, 3=mastery)

    Returns:
        WPM value
    """
    if phase == 1:
        # Phase 1: Foundation (12-25 WPM, moderate only)
        return np.random.uniform(12, 25)
    elif phase == 2:
        # Phase 2: Expansion (8-35 WPM, wider range)
        return np.random.uniform(8, 35)
    else:
        # Phase 3: Mastery (5-40 WPM, full range)
        # Distribution: 10% 5-10, 30% 10-15, 40% 15-25, 15% 25-35, 5% 35-40
        choice = np.random.random()
        if choice < 0.10:
            return np.random.uniform(5, 10)
        elif choice < 0.40:
            return np.random.uniform(10, 15)
        elif choice < 0.80:
            return np.random.uniform(15, 25)
        elif choice < 0.95:
            return np.random.uniform(25, 35)
        else:
            return np.random.uniform(35, 40)
