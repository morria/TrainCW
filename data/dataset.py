"""
PyTorch Dataset classes for CW training data.
"""

import librosa
import numpy as np
import torch
import torch.utils.data

from .generator import generate_training_sample


class CWDataset(torch.utils.data.IterableDataset):
    """
    On-the-fly CW data generation for training.

    This dataset generates infinite synthetic data, ensuring each epoch
    sees new variations. Implements curriculum learning phases.
    """

    def __init__(
        self,
        samples_per_epoch: int = 10000,
        phase: int = 3,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 160,
        seed: int | None = None,
    ):
        """
        Initialize CW dataset.

        Args:
            samples_per_epoch: Number of samples to generate per epoch
            phase: Curriculum phase (1=foundation, 2=expansion, 3=mastery)
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            n_fft: FFT size
            hop_length: Hop length for STFT
            seed: Random seed (None for random)
        """
        super().__init__()
        self.samples_per_epoch = samples_per_epoch
        self.phase = phase
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.seed = seed

        # Character vocabulary
        self.char_to_idx = self._build_vocabulary()
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

    def _build_vocabulary(self) -> dict[str, int]:
        """Build character to index mapping."""
        chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?/ ")
        # Add prosigns
        chars.extend(["<AR>", "<SK>", "<BT>", "<KN>"])
        # Add blank for CTC
        chars.append("<blank>")

        return {char: idx for idx, char in enumerate(chars)}

    def text_to_indices(self, text: str) -> torch.Tensor:
        """
        Convert text to tensor of indices.

        Args:
            text: Input text

        Returns:
            Tensor of character indices
        """
        indices = []
        i = 0
        while i < len(text):
            # Check for 2-char prosigns
            if i < len(text) - 1:
                two_char = text[i : i + 2].upper()
                if f"<{two_char}>" in self.char_to_idx:
                    indices.append(self.char_to_idx[f"<{two_char}>"])
                    i += 2
                    continue

            # Single character
            char = text[i].upper()
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Unknown char - skip or use blank
                pass
            i += 1

        return torch.tensor(indices, dtype=torch.long)

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram from audio.

        Args:
            audio: Input audio waveform

        Returns:
            Mel spectrogram [n_mels, time]
        """
        # Compute mel spectrogram using librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=0,
            fmax=self.sample_rate // 2,
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1]
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (
            mel_spec_db.max() - mel_spec_db.min() + 1e-8
        )

        return mel_spec_normalized

    def __iter__(self):
        """Iterate over dataset, generating samples on-the-fly."""
        # Set worker seed for reproducibility
        if self.seed is not None:
            worker_info = torch.utils.data.get_worker_info()
            seed = self.seed + worker_info.id if worker_info is not None else self.seed
            np.random.seed(seed)

        for _ in range(self.samples_per_epoch):
            # Generate audio sample
            audio, text, metadata = generate_training_sample(
                phase=self.phase, sample_rate=self.sample_rate
            )

            # Compute spectrogram
            spectrogram = self.compute_mel_spectrogram(audio)

            # Convert to tensor [time, n_mels]
            spectrogram_tensor = torch.from_numpy(spectrogram.T).float()

            # Encode text
            text_indices = self.text_to_indices(text)

            # Return (spectrogram, text_indices, text_length, text_string)
            yield {
                "spectrogram": spectrogram_tensor,
                "text_indices": text_indices,
                "text_length": len(text_indices),
                "text": text,
                "metadata": metadata,
            }

    def __len__(self):
        """Return nominal dataset size."""
        return self.samples_per_epoch


class CWTestDataset(torch.utils.data.Dataset):
    """
    Fixed test dataset for evaluation.

    Unlike the training dataset, this loads pre-generated samples
    for consistent evaluation across epochs.
    """

    def __init__(
        self,
        data_path: str,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 160,
    ):
        """
        Initialize test dataset.

        Args:
            data_path: Path to pre-generated test data (.pkl or .pt file)
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            n_fft: FFT size
            hop_length: Hop length
        """
        super().__init__()
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Load data
        self.samples = self._load_data()

        # Build vocabulary (same as training)
        dataset_temp = CWDataset()
        self.char_to_idx = dataset_temp.char_to_idx
        self.idx_to_char = dataset_temp.idx_to_char

    def _load_data(self):
        """Load pre-generated test data."""
        import pickle
        from pathlib import Path

        with Path(self.data_path).open("rb") as f:
            data = pickle.load(f)
        return data

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram (same as training)."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=0,
            fmax=self.sample_rate // 2,
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (
            mel_spec_db.max() - mel_spec_db.min() + 1e-8
        )

        return mel_spec_normalized

    def text_to_indices(self, text: str) -> torch.Tensor:
        """Convert text to indices (same as training)."""
        indices = []
        i = 0
        while i < len(text):
            if i < len(text) - 1:
                two_char = text[i : i + 2].upper()
                if f"<{two_char}>" in self.char_to_idx:
                    indices.append(self.char_to_idx[f"<{two_char}>"])
                    i += 2
                    continue

            char = text[i].upper()
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            i += 1

        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample by index."""
        audio, text, metadata = self.samples[idx]

        # Compute spectrogram
        spectrogram = self.compute_mel_spectrogram(audio)
        spectrogram_tensor = torch.from_numpy(spectrogram.T).float()

        # Encode text
        text_indices = self.text_to_indices(text)

        return {
            "spectrogram": spectrogram_tensor,
            "text_indices": text_indices,
            "text_length": len(text_indices),
            "text": text,
            "metadata": metadata,
        }


def collate_fn(batch):
    """
    Collate function for batching variable-length sequences.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensors with padding
    """
    # Find max lengths
    max_spec_len = max(sample["spectrogram"].shape[0] for sample in batch)
    max_text_len = max(sample["text_length"] for sample in batch)

    batch_size = len(batch)
    n_mels = batch[0]["spectrogram"].shape[1]

    # Prepare padded tensors
    spectrograms = torch.zeros(batch_size, max_spec_len, n_mels)
    spec_lengths = torch.zeros(batch_size, dtype=torch.long)

    text_indices = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    text_lengths = torch.zeros(batch_size, dtype=torch.long)

    texts = []
    metadata_list = []

    for i, sample in enumerate(batch):
        spec = sample["spectrogram"]
        spec_len = spec.shape[0]
        spectrograms[i, :spec_len, :] = spec
        spec_lengths[i] = spec_len

        text_idx = sample["text_indices"]
        text_len = len(text_idx)
        text_indices[i, :text_len] = text_idx
        text_lengths[i] = text_len

        texts.append(sample["text"])
        metadata_list.append(sample["metadata"])

    return {
        "spectrograms": spectrograms,
        "spec_lengths": spec_lengths,
        "text_indices": text_indices,
        "text_lengths": text_lengths,
        "texts": texts,
        "metadata": metadata_list,
    }
