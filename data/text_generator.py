"""
Text content generation for CW training data.
"""

import numpy as np
from typing import List


# Common English words (simplified list)
COMMON_WORDS = [
    'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
    'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW',
    'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID',
    'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'WORK', 'CALL', 'GOOD',
    'HAND', 'HIGH', 'KEEP', 'LAST', 'LEFT', 'LIFE', 'LONG', 'MADE', 'MAKE',
    'MUCH', 'NAME', 'NEVER', 'ONLY', 'OVER', 'PART', 'PLACE', 'RIGHT', 'SAID',
    'SAME', 'SEEM', 'SHOW', 'SMALL', 'SUCH', 'TELL', 'THAN', 'THEM', 'THEN',
    'THERE', 'THESE', 'THEY', 'THING', 'THINK', 'THIS', 'TIME', 'VERY', 'WANT',
    'WELL', 'WENT', 'WHAT', 'WHEN', 'WHERE', 'WHICH', 'WHILE', 'WITH', 'WORD',
    'YEAR', 'YOUR', 'ABOUT', 'AFTER', 'AGAIN', 'BEING', 'BELOW', 'COULD',
    'EVERY', 'FIRST', 'FOUND', 'GREAT', 'HOUSE', 'KNOW', 'LARGE', 'LEARN',
    'LITTLE', 'MIGHT', 'NEVER', 'OTHER', 'PEOPLE', 'SHOULD', 'STILL', 'THEIR',
    'THESE', 'THINK', 'THREE', 'THROUGH', 'UNDER', 'UNTIL', 'WATER', 'WHERE',
    'WORLD', 'WOULD', 'WRITE', 'RADIO', 'SIGNAL', 'GOING', 'STATION', 'ANTENNA',
]

# Ham radio abbreviations
CW_ABBREVIATIONS = [
    'TNX', 'OM', 'YL', 'FB', 'HI', 'ES', 'BTU', 'CUAGN', 'CUL', 'GL', 'GN',
    'GM', 'GA', 'GB', 'GE', 'HR', 'HW', 'RIG', 'PSE', 'RST', 'SIG', 'WX',
    'TU', 'UR', 'VY', 'WKD', 'WKG', 'YR', '73', '88', '599', '5NN',
    'QRL', 'QRM', 'QRN', 'QRZ', 'QRV', 'QSB', 'QSL', 'QSO', 'QSY', 'QTH',
]

# Q-codes and common phrases
QSO_PHRASES = [
    'CQ', 'DE', 'K', 'AR', 'SK', 'BT', 'KN', 'AS', 'HH', 'VE',
    'CQ CQ DE', 'PSE K', 'TU', 'TKS', 'TNX FER', 'QSL?', 'QRZ?',
    'UR RST', 'MY RST', 'ES', 'NAME', 'QTH', 'RIG', 'ANT',
]

# Alphabet for random character generation
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Call sign prefixes
CALLSIGN_PREFIXES = [
    'W', 'K', 'N', 'A',  # USA
    'VE', 'VA',  # Canada
    'G', 'M',  # UK
    'DL', 'DA',  # Germany
    'F',  # France
    'I',  # Italy
    'JA', 'JH', 'JR',  # Japan
    'VK',  # Australia
    'ZL',  # New Zealand
]


class TextGenerator:
    """
    Generate various types of text content for CW training.
    """

    def __init__(self):
        self.words = COMMON_WORDS
        self.abbreviations = CW_ABBREVIATIONS
        self.qso_phrases = QSO_PHRASES

    def generate_random_characters(self, length: int) -> str:
        """Generate random character sequence."""
        return ''.join(np.random.choice(list(ALPHABET), size=length))

    def generate_random_words(self, n_words: int) -> str:
        """Generate random words from dictionary."""
        words = np.random.choice(self.words, size=n_words, replace=True)
        return ' '.join(words)

    def generate_callsign(self) -> str:
        """Generate realistic amateur radio callsign."""
        prefix = np.random.choice(CALLSIGN_PREFIXES)
        number = np.random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        suffix_length = np.random.choice([1, 2, 3], p=[0.1, 0.3, 0.6])
        suffix = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                                         size=suffix_length))
        return f"{prefix}{number}{suffix}"

    def generate_signal_report(self) -> str:
        """Generate RST signal report."""
        r = np.random.choice(['3', '4', '5'])  # Readability
        s = np.random.choice(['3', '4', '5', '6', '7', '8', '9'])  # Strength
        t = np.random.choice(['7', '8', '9'])  # Tone
        return f"{r}{s}{t}"

    def generate_number_sequence(self) -> str:
        """Generate random number sequence."""
        length = np.random.randint(2, 6)
        return ''.join(np.random.choice(list('0123456789'), size=length))

    def generate_abbreviation_sequence(self) -> str:
        """Generate sequence of CW abbreviations."""
        n_abbrev = np.random.randint(2, 5)
        abbrevs = np.random.choice(self.abbreviations, size=n_abbrev, replace=True)
        return ' '.join(abbrevs)

    def generate_qso_exchange(self) -> str:
        """Generate realistic QSO exchange."""
        templates = [
            f"CQ CQ DE {self.generate_callsign()} K",
            f"{self.generate_callsign()} DE {self.generate_callsign()} K",
            f"UR RST {self.generate_signal_report()} {self.generate_signal_report()}",
            f"TNX FER QSO 73 {self.generate_callsign()} DE {self.generate_callsign()} SK",
            f"QRZ DE {self.generate_callsign()}",
        ]
        return np.random.choice(templates)

    def generate_content(self, content_type: str, target_length: int = None) -> str:
        """
        Generate content of specified type.

        Args:
            content_type: One of 'random_chars', 'random_words', 'callsigns',
                         'numbers', 'abbreviations', 'qso'
            target_length: Target length in characters (approximate)

        Returns:
            Generated text
        """
        if content_type == 'random_chars':
            length = target_length or np.random.randint(5, 30)
            # Add spaces for word breaks
            text = self.generate_random_characters(length)
            # Insert spaces randomly
            chars = list(text)
            for i in range(len(chars) // 5):
                pos = np.random.randint(3, len(chars) - 3)
                chars.insert(pos, ' ')
            return ''.join(chars)

        elif content_type == 'random_words':
            n_words = (target_length // 5) if target_length else np.random.randint(3, 8)
            return self.generate_random_words(n_words)

        elif content_type == 'callsigns':
            n_callsigns = np.random.randint(2, 4)
            callsigns = [self.generate_callsign() for _ in range(n_callsigns)]
            # Add DE (from) between callsigns sometimes
            if len(callsigns) >= 2 and np.random.random() < 0.5:
                callsigns.insert(1, 'DE')
            return ' '.join(callsigns)

        elif content_type == 'numbers':
            parts = []
            for _ in range(np.random.randint(2, 4)):
                if np.random.random() < 0.6:
                    parts.append(self.generate_signal_report())
                else:
                    parts.append(self.generate_number_sequence())
            return ' '.join(parts)

        elif content_type == 'abbreviations':
            return self.generate_abbreviation_sequence()

        elif content_type == 'qso':
            return self.generate_qso_exchange()

        else:
            raise ValueError(f"Unknown content type: {content_type}")


def generate_random_text(length: int = None) -> str:
    """
    Generate random text for training.

    Args:
        length: Approximate target length in characters

    Returns:
        Generated text
    """
    generator = TextGenerator()

    # Select content type based on distribution from training plan
    content_types = ['random_chars', 'random_words', 'callsigns',
                    'numbers', 'abbreviations', 'qso']
    weights = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]

    content_type = np.random.choice(content_types, p=weights)

    return generator.generate_content(content_type, length)


def generate_callsign() -> str:
    """Generate a random callsign."""
    generator = TextGenerator()
    return generator.generate_callsign()


def sample_text_length() -> int:
    """
    Sample text length from distribution.

    Returns length in characters, constrained for 2-second windows.
    """
    # Distribution: 10% very short, 30% short, 40% medium, 20% full
    choice = np.random.random()
    if choice < 0.10:
        return np.random.randint(1, 5)    # Very short (1-5 chars)
    elif choice < 0.40:
        return np.random.randint(6, 10)   # Short (6-10 chars)
    elif choice < 0.80:
        return np.random.randint(11, 20)  # Medium (11-20 chars)
    else:
        return np.random.randint(21, 30)  # Full length (21-30 chars)
