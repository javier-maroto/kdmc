import torch.nn.functional as F


class FFloat(object):
    """Class to translate between strings of floats and fractions into float objects.
    """
    def __new__(cls, value) -> float:
        if not isinstance(value, str):
            return float(value)
        return cls.process_str(value)
    
    @classmethod
    def process_str(cls, value) -> float:
        s_list = value.split('/')
        if len(s_list) == 1:
            return float(s_list[0])
        elif len(s_list) == 2:
            return float(s_list[0])/float(s_list[1])
        else:
            raise ValueError('Not a valid number')


def softXEnt(input, target):
    logprobs = F.log_softmax(input, dim=-1)
    return -(target * logprobs).sum() / input.shape[0]


def _reporthook(t):
    """``reporthook`` to use with ``urllib.request`` that prints the process of the download.

    Uses ``tqdm`` for progress bar.

    **Reference:**
    https://github.com/tqdm/tqdm

    Args:
        t (tqdm.tqdm) Progress bar.

    Example:
        >>> with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:  # doctest: +SKIP
        ...   urllib.request.urlretrieve(file_url, filename=full_path, reporthook=reporthook(t))
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        Args:
            b (int, optional): Number of blocks just transferred [default: 1].
            bsize (int, optional): Size of each block (in tqdm units) [default: 1].
            tsize (int, optional): Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def compute_sample_energy(x):
    """Computes the mean energy of the signal (or batch of signals) samples

    Args:
        x (torch.Tensor): IQ signal ('batch', 'time', 'iq')

    Returns:
        torch.Tensor: energy with same number of dimensions, only first of size not 1 (batch)
    """
    return (x ** 2).sum(-2).mean(-1)