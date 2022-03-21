import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flatten the channel, IQ, and time dims into a single feature dim.
    This module assumes that the input signal is structured as:
    .. math::
        Batch x Channel x IQ x Time
    Args:
        preserve_time (bool, optional): If provided as True then the time dimension is
                                        preserved in the outputs and only the IQ and
                                        Channel dimensions are concatenated together.
                                        Otherwise, the time dimension is also collapsed
                                        to form a single feature dimension.  Generally,
                                        you will set this to False if the layer after
                                        Flatten will be a Linear layer and set this to
                                        True if the layer after Flatten will be a
                                        Recurrent layer that utilizes the time
                                        dimension.  Defaults to False.
    The outputs of this layer, if *preserve_time* is not set to True, are:
    .. math::
        Batch x Features
    Where features is the product of the flattened dimensions:
    .. math::
        (Channel x IQ x Time)
    The outputs of this layer, if *preserve_time* is set to True, are:
    .. math::
        Batch x Time x Features
    Where features is the product of the flattened dimensions:
    .. math::
        (Channel x IQ)
    """

    def __init__(self, preserve_time: bool = False):
        super().__init__()
        self._preserve_time = preserve_time

    def forward(self, x: torch.Tensor):
        """Forward method."""
        if self._preserve_time:
            return self._flatten_preserve_time(x=x)
        else:
            return self._flatten(x=x)

    def _flatten(self, x: torch.Tensor):
        if len(x.shape) < 2:
            raise ValueError(
                "The inputs to the Flatten layer must have at least 2 dimensions (e.g. "
                "BxCxIQxT), input shape was {}".format(x.shape)
            )
        # It doesn't entirely matter how many dimensions are in the input or if it is
        # properly structured as 'complex valued' (IQ dimension has 2 values).
        # Therefore, the code to implement this is more general while leaving the
        # docstring more explicit to avoid confusing a caller reading the documentation.
        x = x.contiguous()
        x = x.view(x.size()[0], -1)
        return x

    def _flatten_preserve_time(self, x: torch.Tensor):
        if len(x.shape) != 4:
            raise ValueError(
                "The inputs to the Flatten layer must have at least 4 dimensions (e.g. "
                "BxCxIQxT), input shape was {}".format(x.shape)
            )
        channel_dim, time_dim = 1, 3

        # BxCxIQxT
        x = x.transpose(channel_dim, time_dim)
        # BxTxCxIQ -- Can now collapse the final two dimensions
        x = x.contiguous()
        x = x.view(x.size()[0], x.size()[1], -1)
        return x


class ResidualUnit(nn.Module):
    """ResidualUnit layer used in O'Shea ResNet model. Proposed in: “Over the Air Deep
    Learning Based Radio Signal Classification”: https://arxiv.org/pdf/1712.04578.pdf.

    Since the detailed implementation is unknown, the details are inspired from the lts4 CIFAR10
    implementation of a BasicBlock layer."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        """Forward method."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out


class ResidualStack(nn.Module):
    """ResidualStack layer used in O'Shea ResNet model. Proposed in: “Over the Air Deep
    Learning Based Radio Signal Classification”: https://arxiv.org/pdf/1712.04578.pdf.

    Since the detailed implementation is unknown, the details are inspired from the lts4 CIFAR10
    implementation of a BasicBlock layer."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.ru1 = ResidualUnit(out_ch, out_ch)
        self.ru2 = ResidualUnit(out_ch, out_ch)

    def forward(self, x):
        """Forward method."""
        out = self.bn1(self.conv1(x))
        out = self.ru1(out)
        out = self.ru2(out)
        out = F.max_pool1d(out, 2)
        return out


class ResNet_OShea(nn.Module):
    """ResNet model proposed by O'Shea in “Over the Air Deep Learning Based Radio Signal
    Classification”: https://arxiv.org/pdf/1712.04578.pdf.

    Some implementation choices:
        - AlphaDropout p=0.1:
            0.05 and 0.1 seem to be good values, see subsection "New dropout technique"
            in https://arxiv.org/pdf/1706.02515.pdf

    It is designed to work for inputs with 1024 time samples.

    Expected input: Batch_size x IQ_channel x time_samples
    """

    def __init__(self, num_classes, time_samples=1024):
        super().__init__()
        self.rs1 = ResidualStack(2, 32)  # Out: B x 32 x 512
        self.rs2 = ResidualStack(32, 32)  # Out: B x 32 x 256
        self.rs3 = ResidualStack(32, 32)  # Out: B x 32 x 128
        self.rs4 = ResidualStack(32, 32)  # Out: B x 32 x 64
        self.rs5 = ResidualStack(32, 32)  # Out: B x 32 x 32
        self.rs6 = ResidualStack(32, 32)  # Out: B x 32 x 16
        self.fc1 = nn.Linear(32 * time_samples // 2 ** 6, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.ad1 = nn.AlphaDropout(p=0.1)
        self.ad2 = nn.AlphaDropout(p=0.1)
        self.flatten = Flatten()

    def forward(self, x):
        """Forward method."""
        x = self.rs1(x)
        x = self.rs2(x)
        x = self.rs3(x)
        x = self.rs4(x)
        x = self.rs5(x)
        x = self.rs6(x)
        x = self.flatten(x)
        x = self.ad1(F.selu(self.fc1(x)))
        x = self.ad2(F.selu(self.fc2(x)))
        out = self.fc3(x)
        return out