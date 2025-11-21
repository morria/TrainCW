"""Evaluation metrics for CW decoding.

This module provides:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Edit distance (Levenshtein distance)
- Confusion matrix generation
"""

from collections import defaultdict
from typing import Optional


def edit_distance(s1: str, s2: str) -> int:
    """
    Compute edit distance (Levenshtein distance) between two strings.

    This is the minimum number of single-character edits (insertions,
    deletions, or substitutions) required to change one string into another.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance

    Example:
        >>> edit_distance("CQ", "CQD")
        1  # One insertion
        >>> edit_distance("PARIS", "PARIS")
        0  # Identical
    """
    len1, len2 = len(s1), len(s2)

    # Create DP table
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Initialize base cases
    for i in range(len1 + 1):
        dp[i][0] = i  # Delete all characters from s1
    for j in range(len2 + 1):
        dp[0][j] = j  # Insert all characters from s2

    # Fill DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                # Characters match, no operation needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Take minimum of insert, delete, or substitute
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # Delete
                    dp[i][j - 1],  # Insert
                    dp[i - 1][j - 1],  # Substitute
                )

    return dp[len1][len2]


def calculate_cer(
    predictions: list[str],
    references: list[str],
) -> float:
    """
    Calculate Character Error Rate (CER).

    CER = (Substitutions + Insertions + Deletions) / Total_Characters

    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings

    Returns:
        Character Error Rate (0.0 to 1.0+, lower is better)

    Example:
        >>> predictions = ["CQ CQ", "TEST"]
        >>> references = ["CQ", "TEXT"]
        >>> cer = calculate_cer(predictions, references)
        >>> # CER accounts for extra chars and substitutions
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    total_distance = 0
    total_chars = 0

    for pred, ref in zip(predictions, references):
        distance = edit_distance(pred, ref)
        total_distance += distance
        total_chars += len(ref)

    if total_chars == 0:
        return 0.0 if total_distance == 0 else 1.0

    return total_distance / total_chars


def calculate_wer(
    predictions: list[str],
    references: list[str],
) -> float:
    """
    Calculate Word Error Rate (WER).

    WER = (Word_Substitutions + Word_Insertions + Word_Deletions) / Total_Words

    Words are separated by spaces.

    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings

    Returns:
        Word Error Rate (0.0 to 1.0+, lower is better)

    Example:
        >>> predictions = ["CQ DE W1ABC"]
        >>> references = ["CQ DE K2XYZ"]
        >>> wer = calculate_wer(predictions, references)
        >>> # One word substitution out of 3 words = 0.333...
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    total_distance = 0
    total_words = 0

    for pred, ref in zip(predictions, references):
        # Split into words
        pred_words = pred.split()
        ref_words = ref.split()

        # Calculate word-level edit distance
        distance = _word_edit_distance(pred_words, ref_words)
        total_distance += distance
        total_words += len(ref_words)

    if total_words == 0:
        return 0.0 if total_distance == 0 else 1.0

    return total_distance / total_words


def _word_edit_distance(words1: list[str], words2: list[str]) -> int:
    """
    Calculate edit distance at word level.

    Args:
        words1: List of words from first sentence
        words2: List of words from second sentence

    Returns:
        Word-level edit distance
    """
    len1, len2 = len(words1), len(words2)

    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len1][len2]


def confusion_matrix(
    predictions: list[str],
    references: list[str],
) -> dict[tuple[str, str], int]:
    """
    Generate character-level confusion matrix.

    Tracks which characters are confused with which other characters,
    including insertions and deletions.

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        Dictionary mapping (reference_char, predicted_char) to count
        Special keys: ("<INS>", char) for insertions, (char, "<DEL>") for deletions

    Example:
        >>> predictions = ["E"]
        >>> references = ["T"]
        >>> cm = confusion_matrix(predictions, references)
        >>> cm[("T", "E")]  # T confused as E
        1
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    matrix = defaultdict(int)

    for pred, ref in zip(predictions, references):
        # Get character-level alignment using edit operations
        operations = _get_edit_operations(ref, pred)

        for op_type, ref_char, pred_char in operations:
            if op_type == "correct":
                matrix[(ref_char, ref_char)] += 1
            elif op_type == "substitute":
                matrix[(ref_char, pred_char)] += 1
            elif op_type == "insert":
                matrix[("<INS>", pred_char)] += 1
            elif op_type == "delete":
                matrix[(ref_char, "<DEL>")] += 1

    return dict(matrix)


def _get_edit_operations(
    s1: str, s2: str
) -> list[tuple[str, Optional[str], Optional[str]]]:
    """
    Get the sequence of edit operations to transform s1 into s2.

    Returns list of (operation, s1_char, s2_char) tuples.
    Operations: "correct", "substitute", "insert", "delete"
    """
    len1, len2 = len(s1), len(s2)

    # Build DP table
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack to get operations
    operations = []
    i, j = len1, len2

    while i > 0 or j > 0:
        if i == 0:
            # Insert remaining characters from s2
            operations.append(("insert", None, s2[j - 1]))
            j -= 1
        elif j == 0:
            # Delete remaining characters from s1
            operations.append(("delete", s1[i - 1], None))
            i -= 1
        elif s1[i - 1] == s2[j - 1]:
            # Characters match
            operations.append(("correct", s1[i - 1], s2[j - 1]))
            i -= 1
            j -= 1
        else:
            # Choose minimum operation
            delete_cost = dp[i - 1][j]
            insert_cost = dp[i][j - 1]
            substitute_cost = dp[i - 1][j - 1]

            if substitute_cost <= delete_cost and substitute_cost <= insert_cost:
                # Substitute
                operations.append(("substitute", s1[i - 1], s2[j - 1]))
                i -= 1
                j -= 1
            elif delete_cost <= insert_cost:
                # Delete
                operations.append(("delete", s1[i - 1], None))
                i -= 1
            else:
                # Insert
                operations.append(("insert", None, s2[j - 1]))
                j -= 1

    operations.reverse()
    return operations


def batch_evaluate(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Evaluate a batch of predictions against references.

    Computes both CER and WER.

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        Dictionary with "cer" and "wer" keys

    Example:
        >>> predictions = ["CQ CQ", "TEST"]
        >>> references = ["CQ", "TEXT"]
        >>> metrics = batch_evaluate(predictions, references)
        >>> print(f"CER: {metrics['cer']:.2%}, WER: {metrics['wer']:.2%}")
    """
    cer = calculate_cer(predictions, references)
    wer = calculate_wer(predictions, references)

    return {
        "cer": cer,
        "wer": wer,
    }
