"""
Tests for Morse code utilities.
"""

import pytest
from morse.morse_code import MorseCode, MORSE_CODE_DICT
from morse.timing import TimingCalculator, WPM_TO_UNIT_TIME


def test_morse_encoding():
    """Test basic morse code encoding."""
    morse = MorseCode()

    # Test single letter
    assert morse.text_to_morse('A') == ['.-']
    assert morse.text_to_morse('E') == ['.']
    assert morse.text_to_morse('T') == ['-']

    # Test word
    result = morse.text_to_morse('SOS')
    assert result == ['...', '---', '...']

    # Test with space
    result = morse.text_to_morse('HI OK')
    assert result == ['....', '..', ' ', '---', '-.-']


def test_morse_decoding():
    """Test morse code decoding."""
    morse = MorseCode()

    # Test basic decoding
    assert morse.morse_to_text(['.-']) == 'A'
    assert morse.morse_to_text(['...', '---', '...']) == 'SOS'

    # Test with space
    assert morse.morse_to_text(['....', '..', ' ', '---', '-.-']) == 'HI OK'


def test_text_to_elements():
    """Test conversion to element sequence."""
    morse = MorseCode()

    elements = morse.text_to_elements('A')
    # 'A' = '.-' = dit, element_gap, dah
    assert len(elements) == 3
    assert elements[0][0] == 'dit'
    assert elements[1][0] == 'element_gap'
    assert elements[2][0] == 'dah'


def test_wpm_to_unit_time():
    """Test WPM conversion."""
    # At 20 WPM, unit time should be 60ms
    unit_time = WPM_TO_UNIT_TIME(20)
    assert abs(unit_time - 0.060) < 0.001

    # At 10 WPM, unit time should be 120ms
    unit_time = WPM_TO_UNIT_TIME(10)
    assert abs(unit_time - 0.120) < 0.001


def test_timing_calculator():
    """Test timing calculations."""
    timing = TimingCalculator(wpm=20)

    # Check basic timings
    dit_duration = timing.get_dit_duration()
    dah_duration = timing.get_dah_duration()

    # Dah should be 3x dit (approximately)
    ratio = dah_duration / dit_duration
    assert 2.5 < ratio < 3.5  # Allow some variance


def test_timing_sequence():
    """Test timing sequence generation."""
    morse = MorseCode()
    timing = TimingCalculator(wpm=20)

    elements = morse.text_to_elements('E')  # 'E' = '.'
    sequence = timing.get_timing_sequence(elements)

    # Should have one dit
    assert len(sequence) == 1
    assert sequence[0][1] == True  # Tone on

    # Test 'A' = '.-'
    elements = morse.text_to_elements('A')
    sequence = timing.get_timing_sequence(elements)

    # Should have: dit, gap, dah
    assert len(sequence) == 3
    assert sequence[0][1] == True   # Dit on
    assert sequence[1][1] == False  # Gap off
    assert sequence[2][1] == True   # Dah on
