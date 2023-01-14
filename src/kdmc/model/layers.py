"""Normalize the power across each example/channel to 1.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import torch
import torch.nn as nn


def energy_fun(x: torch.Tensor, sps: float = 1.0):
    """Calculate the average energy (per symbol if provided) for each example.
    This function assumes that the signal is structured as:
    .. math::
        Batch x Channel x IQ x Time.
    Args:
        x (torch.Tensor): Input Tensor (BxCxIQxT)
        sps (int, optional): Samples per symbol, essentially the power is multiplied by
                             this value in order to calculate average energy per symbol.
                             Defaults to 1.0.
    .. math::
        \mathbb{E}[E_{s}] = \\frac{\\text{sps}}{N} \sum_{i=0}^{N} |s_i|^2
        |s_i| = \sqrt{\mathbb{R}^2 + \mathbb{C}^2}
    Returns:
        [torch.Tensor]: Average energy per example per channel (BxC)
    """
    if len(x.shape) != 4:
        raise ValueError(
            "The inputs to the energy function must have 4 dimensions (BxCxIQxT), "
            "input shape was {}".format(x.shape)
        )
    if x.shape[2] != 2:
        raise ValueError(
            "The inputs to the energy function must be 'complex valued' by having 2 "
            "elements in the IQ dimension (BxCxIQxT), input shape was {}".format(
                x.shape
            )
        )
    iq_dim = 2
    time_dim = 3

    r, c = x.chunk(chunks=2, dim=iq_dim)
    power = (r * r) + (c * c)  # power is magnitude squared so sqrt cancels

    # pylint: disable=no-member
    # The linter isn't able to find the "mean" function but its there!
    x = torch.mean(power, dim=time_dim) * sps

    # This Tensor still has an unnecessary singleton dimensions in IQ
    x = x.squeeze(dim=iq_dim)

    return x


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


class PowerNormalization(nn.Module):
    """Perform average energy per sample (power) normalization.
    Power Normalization would be performed as follows for each batch/channel:
    .. math::
        x = \\frac{x}{\sqrt{\mathbb{E}[x]}}
    This module assumes that the signal is structured as:
    .. math::
        Batch x Channel x IQ x Time.
    Where the power normalization is performed along the T axis using the power
    measured in the complex valued I/Q dimension.
    The outputs of this layer match the inputs:
    .. math::
        Batch x Channel x IQ x Time
    """

    def forward(self, x: torch.Tensor):
        if len(x.shape) != 4:
            raise ValueError(
                "The inputs to the PowerNormalization layer must have 4 dimensions "
                "(BxCxIQxT), input shape was {}".format(x.shape)
            )
        if x.shape[2] != 2:
            raise ValueError(
                "The inputs to the PowerNormalization layer must be 'complex valued' "
                "by having 2 elements in the IQ dimension (BxCxIQxT), input shape was "
                "{}".format(x.shape)
            )

        energy = energy_fun(x)
        # Make the dimensions match because broadcasting is too magical to
        # understand in its entirety... essentially want to ensure that we
        # divide each channel of each example by the sqrt of the power of
        # that channel/example pair
        energy = energy.view([energy.size()[0], energy.size()[1], 1, 1])

        return x / torch.sqrt(energy)