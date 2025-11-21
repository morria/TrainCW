"""CNN encoder for spectrogram feature extraction."""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    CNN encoder for extracting features from mel-spectrograms.

    This encoder uses a series of convolutional blocks to extract
    time-frequency features from the input spectrogram. Each block
    consists of Conv2D -> BatchNorm -> ReLU -> MaxPool.

    The output is reshaped to a sequence suitable for LSTM processing.

    Args:
        input_channels: Number of input channels (default: 1 for mono spectrogram)
        cnn_channels: List of output channels for each conv layer
        kernel_sizes: List of kernel sizes for each conv layer
        strides: List of strides for each conv layer
        pooling: List of pooling sizes for each conv layer

    Input shape:
        (batch, 1, n_mels, time)

    Output shape:
        (batch, time', features) where time' is reduced by pooling
    """

    def __init__(
        self,
        input_channels: int = 1,
        cnn_channels: list[int] = [32, 64, 128, 256],
        kernel_sizes: list[int] = [3, 3, 3, 3],
        strides: list[int] = [1, 1, 1, 1],
        pooling: list[int] = [2, 2, 2, 2],
    ):
        super().__init__()

        assert len(cnn_channels) == len(kernel_sizes) == len(strides) == len(pooling), (
            "All layer configuration lists must have the same length"
        )

        self.input_channels = input_channels
        self.cnn_channels = cnn_channels

        # Build convolutional blocks
        layers = []
        in_channels = input_channels

        for out_channels, kernel_size, stride, pool_size in zip(
            cnn_channels, kernel_sizes, strides, pooling
        ):
            # Conv2D block
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,  # 'same' padding
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            # Max pooling (frequency and time)
            if pool_size > 1:
                layers.append(nn.MaxPool2d(kernel_size=pool_size))

            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        # Calculate output feature dimension
        # After all pooling, the frequency dimension is reduced
        # We'll flatten frequency × channels to get final feature dimension
        self._output_feature_dim = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN encoder.

        Args:
            x: Input tensor of shape (batch, n_mels, time) or (batch, 1, n_mels, time)

        Returns:
            Output tensor of shape (batch, time', features)
        """
        # Ensure 4D input: (batch, channels, freq, time)
        if x.ndim == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        # Apply CNN
        x = self.cnn(x)  # (batch, channels, freq', time')

        # Reshape for LSTM: (batch, time', channels * freq')
        batch_size, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)  # (batch, time', channels, freq')
        x = x.reshape(batch_size, time, channels * freq)  # Flatten freq and channels

        return x

    def get_output_dim(self, input_freq_dim: int) -> int:
        """
        Calculate output feature dimension given input frequency dimension.

        Args:
            input_freq_dim: Input frequency dimension (e.g., n_mels=64)

        Returns:
            Output feature dimension (channels * reduced_freq)
        """
        if self._output_feature_dim is not None:
            return self._output_feature_dim

        # Simulate forward pass to get output dimension
        dummy_input = torch.zeros(1, 1, input_freq_dim, 100)
        with torch.no_grad():
            output = self.cnn(dummy_input)
        _, channels, freq, _ = output.shape

        self._output_feature_dim = channels * freq
        return self._output_feature_dim
