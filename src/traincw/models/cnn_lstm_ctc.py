"""Complete CNN-LSTM-CTC model for CW decoding."""

import torch
import torch.nn as nn

from traincw.models.decoder import BLANK_IDX, decode_predictions, get_vocabulary_size
from traincw.models.encoder import CNNEncoder


class CWDecoder(nn.Module):
    """
    Complete CNN-LSTM-CTC model for Morse code decoding.

    Architecture:
        Input: Mel-spectrogram (batch, n_mels, time)
            ↓
        CNN Encoder: Extract time-frequency features
            ↓
        Bidirectional LSTM: Temporal modeling
            ↓
        Fully Connected: Map to character probabilities
            ↓
        Output: Character logits (batch, time', num_classes)

    The model is trained with CTC loss and can be decoded using
    greedy or beam search decoding.

    Args:
        n_mels: Number of mel frequency bins (default: 64)
        cnn_channels: CNN output channels per layer
        cnn_kernel_sizes: CNN kernel sizes per layer
        cnn_strides: CNN strides per layer
        cnn_pooling: CNN pooling sizes per layer
        lstm_hidden_size: LSTM hidden size (default: 256)
        lstm_num_layers: Number of LSTM layers (default: 2)
        lstm_dropout: LSTM dropout rate (default: 0.1)
        bidirectional: Use bidirectional LSTM (default: True)
        num_classes: Number of output classes (default: 45)
    """

    def __init__(
        self,
        n_mels: int = 64,
        cnn_channels: list[int] = [32, 64, 128, 256],
        cnn_kernel_sizes: list[int] = [3, 3, 3, 3],
        cnn_strides: list[int] = [1, 1, 1, 1],
        cnn_pooling: list[int] = [2, 2, 2, 2],
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.1,
        bidirectional: bool = True,
        num_classes: int = 45,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        # CNN encoder
        self.encoder = CNNEncoder(
            input_channels=1,
            cnn_channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            pooling=cnn_pooling,
        )

        # Get CNN output dimension
        cnn_output_dim = self.encoder.get_output_dim(n_mels)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Fully connected output layer
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input spectrogram of shape (batch, n_mels, time) or (batch, 1, n_mels, time)

        Returns:
            Character logits of shape (batch, time', num_classes)
        """
        # CNN encoder
        x = self.encoder(x)  # (batch, time', cnn_features)

        # LSTM
        x, _ = self.lstm(x)  # (batch, time', lstm_hidden * 2)

        # Fully connected
        x = self.fc(x)  # (batch, time', num_classes)

        return x

    def decode(
        self,
        spectrograms: torch.Tensor,
        method: str = "greedy",
        beam_width: int = 10,
    ) -> list[str]:
        """
        Decode spectrograms to text.

        Args:
            spectrograms: Input spectrograms (batch, n_mels, time)
            method: Decoding method, "greedy" or "beam_search"
            beam_width: Beam width for beam search

        Returns:
            List of decoded text strings
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(spectrograms)
            texts = decode_predictions(logits, method=method, beam_width=beam_width)
        return texts

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Get model size in megabytes (float32)."""
        num_params = self.get_num_parameters()
        size_mb = (num_params * 4) / (1024**2)  # 4 bytes per float32
        return size_mb


def create_model_from_config(config) -> CWDecoder:
    """
    Create CWDecoder model from configuration.

    Args:
        config: Configuration object with model and audio settings

    Returns:
        Initialized CWDecoder model
    """
    model = CWDecoder(
        n_mels=config.audio.n_mels,
        cnn_channels=config.model.cnn_channels,
        cnn_kernel_sizes=config.model.cnn_kernel_sizes,
        cnn_strides=config.model.cnn_strides,
        cnn_pooling=config.model.cnn_pooling,
        lstm_hidden_size=config.model.lstm_hidden_size,
        lstm_num_layers=config.model.lstm_num_layers,
        lstm_dropout=config.model.lstm_dropout,
        bidirectional=config.model.bidirectional,
        num_classes=config.model.num_classes,
    )

    return model


def compute_ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = BLANK_IDX,
) -> torch.Tensor:
    """
    Compute CTC loss.

    Args:
        logits: Model output of shape (batch, time, num_classes)
        targets: Target indices of shape (batch, max_target_length)
        input_lengths: Lengths of input sequences (batch,)
        target_lengths: Lengths of target sequences (batch,)
        blank: Index of blank token

    Returns:
        CTC loss (scalar)
    """
    # CTC loss expects log probabilities
    log_probs = torch.log_softmax(logits, dim=2)

    # CTC expects (time, batch, num_classes)
    log_probs = log_probs.permute(1, 0, 2)

    # Flatten targets for CTC loss
    targets_flat = []
    for i, length in enumerate(target_lengths):
        targets_flat.extend(targets[i, :length].tolist())
    targets_flat = torch.tensor(targets_flat, dtype=torch.long, device=logits.device)

    # Compute CTC loss
    loss = nn.functional.ctc_loss(
        log_probs=log_probs,
        targets=targets_flat,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=blank,
        reduction="mean",
        zero_infinity=True,  # Handle infinite losses
    )

    return loss
