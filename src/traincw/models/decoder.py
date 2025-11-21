"""CTC decoder utilities for inference."""

import torch


# Character vocabulary for CW decoding
# Includes: A-Z, 0-9, punctuation, prosigns
CHAR_VOCABULARY = [
    " ",  # Space (word separator)
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ".",
    ",",
    "?",
    "/",
    "<AR>",
    "<SK>",
    "<BT>",
    "<KN>",  # Prosigns
]

# Create mapping dictionaries
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHAR_VOCABULARY)}  # +1 for blank
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHAR_VOCABULARY)}
BLANK_IDX = 0  # CTC blank token
IDX_TO_CHAR[BLANK_IDX] = "<BLANK>"


def text_to_indices(text: str) -> list[int]:
    """
    Convert text string to list of character indices.

    Args:
        text: Input text string

    Returns:
        List of character indices

    Example:
        >>> text_to_indices("CQ")
        [3, 17]  # Indices for 'C' and 'Q'
    """
    indices = []
    i = 0
    text = text.upper()

    while i < len(text):
        # Check for prosigns (multi-character tokens)
        found_prosign = False
        for prosign in ["<AR>", "<SK>", "<BT>", "<KN>"]:
            if text[i:].startswith(prosign):
                indices.append(CHAR_TO_IDX[prosign])
                i += len(prosign)
                found_prosign = True
                break

        if not found_prosign:
            char = text[i]
            if char in CHAR_TO_IDX:
                indices.append(CHAR_TO_IDX[char])
            else:
                # Unknown character - skip or replace with space
                if char.strip():  # Non-whitespace unknown char
                    print(f"Warning: Unknown character '{char}' skipped")
            i += 1

    return indices


def indices_to_text(indices: list[int]) -> str:
    """
    Convert list of character indices to text string.

    Args:
        indices: List of character indices

    Returns:
        Text string

    Example:
        >>> indices_to_text([3, 17])
        'CQ'
    """
    chars = []
    for idx in indices:
        if idx in IDX_TO_CHAR:
            char = IDX_TO_CHAR[idx]
            if char != "<BLANK>":
                chars.append(char)
        else:
            print(f"Warning: Unknown index {idx} skipped")

    return "".join(chars)


def ctc_greedy_decode(logits: torch.Tensor, blank: int = 0) -> list[list[int]]:
    """
    Greedy CTC decoding (best path decoding).

    Takes the most likely character at each time step and removes
    blanks and repeated characters according to CTC rules.

    Args:
        logits: Output logits from model, shape (batch, time, num_classes)
        blank: Index of blank token (default: 0)

    Returns:
        List of decoded sequences (one per batch item), each a list of indices

    Example:
        >>> logits = torch.randn(2, 100, 45)  # batch=2, time=100, classes=45
        >>> decoded = ctc_greedy_decode(logits)
        >>> len(decoded)
        2
    """
    # Get most likely class at each time step
    _, max_indices = torch.max(logits, dim=2)  # (batch, time)

    decoded_sequences = []

    for batch_idx in range(max_indices.size(0)):
        indices = max_indices[batch_idx].tolist()

        # Remove consecutive duplicates and blanks
        decoded = []
        previous = None

        for idx in indices:
            if idx != blank and idx != previous:
                decoded.append(idx)
            previous = idx

        decoded_sequences.append(decoded)

    return decoded_sequences


def ctc_beam_search_decode(
    logits: torch.Tensor,
    beam_width: int = 10,
    blank: int = 0,
) -> list[tuple[list[int], float]]:
    """
    Beam search CTC decoding.

    More accurate than greedy decoding but slower. Maintains top-k
    candidate sequences at each time step.

    Args:
        logits: Output logits from model, shape (batch, time, num_classes)
        beam_width: Number of beams to keep (default: 10)
        blank: Index of blank token (default: 0)

    Returns:
        List of (decoded_sequence, score) tuples (one per batch item)

    Note:
        This is a simplified beam search. For production, consider using
        a more sophisticated implementation with language model integration.
    """
    log_probs = torch.log_softmax(logits, dim=2)  # (batch, time, num_classes)

    batch_size, time_steps, num_classes = log_probs.shape
    results = []

    for batch_idx in range(batch_size):
        batch_log_probs = log_probs[batch_idx]  # (time, num_classes)

        # Initialize beams: (prefix, prob)
        beams = {(): 0.0}  # Empty prefix with probability 1.0 (log=0)

        for t in range(time_steps):
            new_beams = {}

            for prefix, prefix_prob in beams.items():
                for c in range(num_classes):
                    char_prob = batch_log_probs[t, c].item()

                    if c == blank:
                        # Blank: extend same prefix
                        new_prefix = prefix
                    else:
                        # Non-blank: add character (avoid consecutive duplicates)
                        # Same character: blank must separate them in CTC
                        new_prefix = prefix if len(prefix) > 0 and prefix[-1] == c else (*prefix, c)

                    # Update beam probability
                    new_prob = prefix_prob + char_prob
                    if new_prefix not in new_beams or new_prob > new_beams[new_prefix]:
                        new_beams[new_prefix] = new_prob

            # Keep top-k beams
            beams = dict(sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width])

        # Get best beam
        if beams:
            best_prefix, best_score = max(beams.items(), key=lambda x: x[1])
            results.append((list(best_prefix), best_score))
        else:
            results.append(([], float("-inf")))

    return results


def decode_predictions(
    logits: torch.Tensor,
    method: str = "greedy",
    beam_width: int = 10,
) -> list[str]:
    """
    Decode model predictions to text strings.

    Args:
        logits: Output logits from model, shape (batch, time, num_classes)
        method: Decoding method, "greedy" or "beam_search" (default: "greedy")
        beam_width: Beam width for beam search (default: 10)

    Returns:
        List of decoded text strings (one per batch item)

    Example:
        >>> logits = torch.randn(2, 100, 45)
        >>> texts = decode_predictions(logits, method="greedy")
        >>> print(texts[0])
        'CQ CQ DE W1ABC'
    """
    if method == "greedy":
        decoded_indices = ctc_greedy_decode(logits)
        return [indices_to_text(indices) for indices in decoded_indices]
    elif method == "beam_search":
        results = ctc_beam_search_decode(logits, beam_width=beam_width)
        return [indices_to_text(indices) for indices, _ in results]
    else:
        raise ValueError(f"Unknown decoding method: {method}")


def get_vocabulary_size() -> int:
    """
    Get the vocabulary size (number of classes including blank).

    Returns:
        Vocabulary size
    """
    return len(CHAR_VOCABULARY) + 1  # +1 for blank


def get_vocabulary() -> list[str]:
    """
    Get the full character vocabulary.

    Returns:
        List of characters in vocabulary
    """
    return CHAR_VOCABULARY.copy()
