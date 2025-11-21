"""
Morse code encoding and decoding utilities.
"""

# International Morse Code dictionary
MORSE_CODE_DICT = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    ".": ".-.-.-",
    ",": "--..--",
    "?": "..--..",
    "/": "-..-.",
    # Prosigns (sent as single character without gap)
    "AR": ".-.-.",  # End of message (also period)
    "SK": "...-.-",  # End of contact
    "BT": "-...-",  # Break/Pause
    "KN": "-.--.",  # Invitation to named station only
}

# Reverse mapping for decoding
MORSE_TO_CHAR = {v: k for k, v in MORSE_CODE_DICT.items()}


class MorseCode:
    """
    Morse code encoder/decoder with timing information.
    """

    def __init__(self):
        self.code_dict = MORSE_CODE_DICT
        self.reverse_dict = MORSE_TO_CHAR

    def text_to_morse(self, text: str) -> list[str]:
        """
        Convert text to morse code patterns.

        Args:
            text: Input text (letters, numbers, punctuation)

        Returns:
            List of morse patterns (e.g., ['.-', '-...'])
        """
        text = text.upper()
        morse_patterns = []

        i = 0
        while i < len(text):
            # Check for prosigns (2-char sequences)
            if i < len(text) - 1:
                two_char = text[i : i + 2]
                if two_char in self.code_dict:
                    morse_patterns.append(self.code_dict[two_char])
                    i += 2
                    continue

            # Single character
            char = text[i]
            if char == " ":
                morse_patterns.append(" ")  # Word space
            elif char in self.code_dict:
                morse_patterns.append(self.code_dict[char])
            else:
                # Skip unknown characters
                pass
            i += 1

        return morse_patterns

    def morse_to_text(self, morse_patterns: list[str]) -> str:
        """
        Decode morse patterns back to text.

        Args:
            morse_patterns: List of morse patterns

        Returns:
            Decoded text
        """
        text = []
        for pattern in morse_patterns:
            if pattern == " ":
                text.append(" ")
            elif pattern in self.reverse_dict:
                text.append(self.reverse_dict[pattern])
            else:
                text.append("?")  # Unknown pattern

        return "".join(text)

    def text_to_elements(self, text: str) -> list[tuple[str, str]]:
        """
        Convert text to a sequence of morse elements with labels.

        Args:
            text: Input text

        Returns:
            List of (element_type, character) tuples where element_type is:
            - 'dit': Short mark
            - 'dah': Long mark
            - 'element_gap': Gap within character
            - 'char_gap': Gap between characters
            - 'word_gap': Gap between words
        """
        morse_patterns = self.text_to_morse(text)
        elements = []

        for i, pattern in enumerate(morse_patterns):
            if pattern == " ":
                elements.append(("word_gap", " "))
            else:
                # Add elements for this character
                for j, symbol in enumerate(pattern):
                    if symbol == ".":
                        elements.append(("dit", text[i] if i < len(text) else "?"))
                    elif symbol == "-":
                        elements.append(("dah", text[i] if i < len(text) else "?"))

                    # Add element gap (except after last element in character)
                    if j < len(pattern) - 1:
                        elements.append(("element_gap", text[i] if i < len(text) else "?"))

                # Add character gap (except after last character)
                if i < len(morse_patterns) - 1 and morse_patterns[i + 1] != " ":
                    elements.append(("char_gap", " "))

        return elements

    def get_character_duration(
        self, char: str, unit_time: float, timing_variance: dict | None = None
    ) -> float:
        """
        Calculate the duration of a character in seconds.

        Args:
            char: Character to encode
            unit_time: Duration of one unit (dit) in seconds
            timing_variance: Optional dict with timing variance parameters

        Returns:
            Duration in seconds
        """
        if char == " ":
            return unit_time * 7  # Word gap

        if char.upper() not in self.code_dict:
            return 0.0

        pattern = self.code_dict[char.upper()]
        duration = 0.0

        for i, symbol in enumerate(pattern):
            if symbol == ".":
                duration += unit_time  # Dit = 1 unit
            elif symbol == "-":
                duration += unit_time * 3  # Dah = 3 units

            # Inter-element gap (except after last element)
            if i < len(pattern) - 1:
                duration += unit_time  # 1 unit gap

        return duration
