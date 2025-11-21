"""Tests for evaluation metrics."""

import pytest

from traincw.evaluation.metrics import (
    batch_evaluate,
    calculate_cer,
    calculate_wer,
    confusion_matrix,
    edit_distance,
)


class TestEditDistance:
    """Test edit distance calculation."""

    def test_identical_strings(self):
        """Test edit distance for identical strings."""
        assert edit_distance("ABC", "ABC") == 0
        assert edit_distance("", "") == 0

    def test_single_insertion(self):
        """Test edit distance for single insertion."""
        assert edit_distance("ABC", "ABCD") == 1

    def test_single_deletion(self):
        """Test edit distance for single deletion."""
        assert edit_distance("ABCD", "ABC") == 1

    def test_single_substitution(self):
        """Test edit distance for single substitution."""
        assert edit_distance("ABC", "ABX") == 1

    def test_empty_strings(self):
        """Test edit distance with empty strings."""
        assert edit_distance("", "ABC") == 3
        assert edit_distance("ABC", "") == 3

    def test_complex_case(self):
        """Test edit distance for complex case."""
        # "PARIS" to "PARES": 1 substitution (I->E)
        assert edit_distance("PARIS", "PARES") == 1

        # "CQ" to "CQD": 1 insertion
        assert edit_distance("CQ", "CQD") == 1


class TestCER:
    """Test Character Error Rate."""

    def test_perfect_predictions(self):
        """Test CER for perfect predictions."""
        predictions = ["ABC", "DEF"]
        references = ["ABC", "DEF"]

        cer = calculate_cer(predictions, references)
        assert cer == 0.0

    def test_single_error(self):
        """Test CER with single error."""
        predictions = ["ABX"]
        references = ["ABC"]

        cer = calculate_cer(predictions, references)
        assert cer == pytest.approx(1.0 / 3.0)  # 1 error in 3 chars

    def test_multiple_samples(self):
        """Test CER with multiple samples."""
        predictions = ["ABC", "XYZ"]
        references = ["ABC", "XY"]

        # First: 0 errors, Second: 1 insertion
        # Total: 1 error / 5 chars = 0.2
        cer = calculate_cer(predictions, references)
        assert cer == pytest.approx(0.2)

    def test_empty_reference(self):
        """Test CER with empty reference."""
        predictions = ["ABC"]
        references = [""]

        # 3 insertions, 0 reference chars -> CER = 1.0 (by convention)
        cer = calculate_cer(predictions, references)
        assert cer == 1.0

    def test_mismatched_lengths(self):
        """Test CER with mismatched prediction/reference lengths."""
        predictions = ["ABC"]
        references = ["ABC", "DEF"]

        with pytest.raises(ValueError):
            calculate_cer(predictions, references)


class TestWER:
    """Test Word Error Rate."""

    def test_perfect_predictions(self):
        """Test WER for perfect predictions."""
        predictions = ["CQ CQ DE W1ABC"]
        references = ["CQ CQ DE W1ABC"]

        wer = calculate_wer(predictions, references)
        assert wer == 0.0

    def test_single_word_error(self):
        """Test WER with single word substitution."""
        predictions = ["CQ DE W1ABC"]
        references = ["CQ DE K2XYZ"]

        # 1 word error (W1ABC -> K2XYZ) out of 3 words
        wer = calculate_wer(predictions, references)
        assert wer == pytest.approx(1.0 / 3.0)

    def test_word_insertion(self):
        """Test WER with word insertion."""
        predictions = ["CQ CQ DE"]
        references = ["CQ DE"]

        # 1 insertion (CQ) out of 2 reference words
        wer = calculate_wer(predictions, references)
        assert wer == pytest.approx(0.5)

    def test_empty_reference(self):
        """Test WER with empty reference."""
        predictions = ["CQ"]
        references = [""]

        wer = calculate_wer(predictions, references)
        assert wer == 1.0


class TestConfusionMatrix:
    """Test confusion matrix generation."""

    def test_correct_predictions(self):
        """Test confusion matrix with correct predictions."""
        predictions = ["ABC"]
        references = ["ABC"]

        cm = confusion_matrix(predictions, references)

        assert cm[("A", "A")] == 1
        assert cm[("B", "B")] == 1
        assert cm[("C", "C")] == 1

    def test_substitution(self):
        """Test confusion matrix with substitution."""
        predictions = ["E"]
        references = ["T"]

        cm = confusion_matrix(predictions, references)

        assert cm[("T", "E")] == 1

    def test_insertion(self):
        """Test confusion matrix with insertion."""
        predictions = ["ABC"]
        references = ["AB"]

        cm = confusion_matrix(predictions, references)

        # Two correct, one insertion
        assert cm[("A", "A")] == 1
        assert cm[("B", "B")] == 1
        assert cm[("<INS>", "C")] == 1

    def test_deletion(self):
        """Test confusion matrix with deletion."""
        predictions = ["AB"]
        references = ["ABC"]

        cm = confusion_matrix(predictions, references)

        # Two correct, one deletion
        assert cm[("A", "A")] == 1
        assert cm[("B", "B")] == 1
        assert cm[("C", "<DEL>")] == 1


class TestBatchEvaluate:
    """Test batch evaluation."""

    def test_batch_evaluate(self):
        """Test batch evaluate returns both CER and WER."""
        predictions = ["CQ CQ", "TEST"]
        references = ["CQ", "TEXT"]

        metrics = batch_evaluate(predictions, references)

        assert "cer" in metrics
        assert "wer" in metrics
        assert isinstance(metrics["cer"], float)
        assert isinstance(metrics["wer"], float)
        assert 0.0 <= metrics["cer"] <= 2.0
        assert 0.0 <= metrics["wer"] <= 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
