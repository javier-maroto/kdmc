import numpy as np
import matplotlib.pyplot as plt


def plot_acc_vs_snr(
        acc_vs_snr, snr, title: str = None, figsize=(20.0, 10.0), annotate: bool = True):
    """Plot Classification Accuracy vs Signal-to-Noise Ratio (SNR).

    Args:
        acc_vs_snr (Iterable[float]): Classification accuracy at each SNR.
        snr (Iterable[float]): Signal-to-Noise Ratios (SNR) that were used for
                               evaluation.
        title (str, optional): Title to put above the plot.  Defaults to None.
        figsize (Tuple[float, float], optional): Size of the figure to create. Defaults
                                                  to (10.0, 5.0).
        annotate (bool, optional): If True then the peak accuracy will be annotated with
                                   a horizontal line and with text describing the value.
                                   If False, no lines or text are added on top of the
                                   plotted data.  Defaults to True.

    Raises:
        ValueError: If the lengths of acc_vs_snr and snr do not match

    Returns:
        Figure: Figure that the results were plotted onto (e.g. for saving plot)
    """
    if len(acc_vs_snr) != len(snr):
        raise ValueError(
            "The lengths of acc_vs_snr and snr must match.  "
            "They were {} and {} respectively.".format(len(acc_vs_snr), len(snr))
        )

    # Sort both arrays by SNR to ensure a smoother line plot
    idxs = np.argsort(snr)
    snr = np.array(snr)[idxs]
    acc_vs_snr = np.array(acc_vs_snr)[idxs]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(snr, acc_vs_snr)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Classification Accuracy")

    if title is not None:
        ax.set_title(title, loc="left", fontweight="bold")

    if annotate:
        peak_acc = np.max(acc_vs_snr)
        ax.axhline(peak_acc, c="k", linestyle="--")
        ax.text(
            x=snr[0] + 0.5,
            y=peak_acc - 0.05,
            s="Peak Classification Accuracy ({:.0f}%)".format(peak_acc * 100),
            bbox=dict(facecolor="white", alpha=0.5),
        )

    return fig


def plot_acc_vs_snr_by_class(
        df_snr, labels, title: str = None, figsize=(20.0, 10.0)):
    """Plot Classification Accuracy vs Signal-to-Noise Ratio (SNR) by class.
    Can plot up to 24 classes without repeating line patterns.

    Returns:
        Figure: Figure that the results were plotted onto (e.g. for saving plot)
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    styles = ['solid', 'dashed', 'dashdot', 'dotted']
    for i, y_class in enumerate(df_snr.y.unique()):
        df_loc = df_snr.loc[df_snr.y == y_class].sort_values('snr')
        ax.plot(df_loc.snr, df_loc.acc, linestyle=styles[i // 6], label=labels[y_class])
    ax.legend()
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Classification Accuracy")

    if title is not None:
        ax.set_title(title, loc="left", fontweight="bold")

    return fig