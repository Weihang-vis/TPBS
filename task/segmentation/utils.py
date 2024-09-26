import json
import os
from numpy import inf, NaN, newaxis, argmin, delete, asarray, isnan, sum, nanmean
from scipy.spatial.distance import cdist
from regional import one, many
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast
import torch
import numpy as np
import torch.sparse
from torch import Tensor
from scipy.ndimage import distance_transform_edt as eucl_distance

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)

'''<<<<<<<<<<<<<<<<<<<<functions from neurofinder>>>>>>>>>>>>>>>>>>>>>>>'''
def load(file):
    """
    Load neuronal regions from a file or string.
    """
    if os.path.isfile(file):
        with open(file, 'r') as f:
            values = json.load(f)
    else:
        values = json.loads(file)

    return many([v['coordinates'] for v in values])


def match(a, b, threshold=inf):
    """
    Find unique matches between two sets of regions.

    Params
    ------
    a, b : regions
        The regions to match.

    threshold : scalar, optional, default = inf
        Threshold distance to use when selecting matches.
    """
    targets = b.center
    target_inds = range(0, len(targets))
    matches = []
    for s in a:
        update = 1

        # skip if no targets left, otherwise update
        if len(targets) == 0:
            update = 0
        else:
            dists = cdist(targets, s.center[newaxis])
            if dists.min() < threshold:
                ind = argmin(dists)
            else:
                update = 0

        # apply updates, otherwise add a nan
        if update == 1:
            matches.append(target_inds[ind])
            targets = delete(targets, ind, axis=0)
            target_inds = delete(target_inds, ind)
        else:
            matches.append(NaN)

    return matches


def shapes(a, b, threshold=inf):
    """
    Compare shapes between two sets of regions.

    Parameters
    ----------
    a, b : regions
        The regions for which to estimate overlap.

    threshold : scalar, optional, default = inf
        Threshold distance to use when matching indices.
    """
    inds = match(a, b, threshold=threshold)
    d = []
    for jj, ii in enumerate(inds):
        if ii is not NaN:
            d.append(a[jj].overlap(b[ii], method='rates'))
        else:
            d.append((NaN, NaN))

    result = asarray(d)

    if sum(~isnan(result)) > 0:
        inclusion, exclusion = tuple(nanmean(result, axis=0))
    else:
        inclusion, exclusion = 0.0, 0.0

    return inclusion, exclusion


def centers(a, b, threshold=inf):
    """
    Compare centers between two sets of regions.

    The recall rate is the number of matches divided by the number in self,
    and the precision rate is the number of matches divided by the number in other.
    Typically a is ground truth and b is an estimate.
    The F score is defined as 2 * (recall * precision) / (recall + precision)

    Before computing metrics, all sources in self are matched to other,
    and a threshold can be set to control matching.

    Parameters
    ----------
    a, b : regions
        The regions for which to estimate overlap.

    threshold : scalar, optional, default = 5
        The distance below which a source is considered found.
    """
    inds = match(a, b, threshold=threshold)

    d = []
    for jj, ii in enumerate(inds):
        if ii is not NaN:
            d.append(a[jj].distance(b[ii]))
        else:
            d.append(NaN)

    result = asarray(d)

    result[isnan(result)] = inf
    compare = lambda x: x < threshold

    recall = sum(asarray(list(map(compare, result)))) / float(a.count)
    precision = sum(asarray(list(map(compare, result)))) / float(b.count)

    return recall, precision


def evaluate(files, threshold=5):
    a = load(files[0])
    b = load(files[1])

    recall, precision = centers(a, b, threshold=threshold)
    inclusion, exclusion = shapes(a, b, threshold=threshold)

    if recall == 0 and precision == 0:
        combined = 0
    else:
        combined = 2 * (recall * precision) / (recall + precision)

    result = {'combined': round(combined, 4), 'inclusion': round(inclusion, 4), 'precision': round(precision, 4),
              'recall': round(recall, 4), 'exclusion': round(exclusion, 4)}
    print(json.dumps(result))


'''<<<<<<<<<<<<<<<<<<functions for boundary loss>>>>>>>>>>>>>>>>>>>>>'''
def id_(x):
    return x

def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def depth(e: List) -> int:
    """
    Compute the depth of nested lists
    """
    if type(e) == list and e:
        return 1 + depth(e[0])

    return 0

def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape  # type: Tuple[int, ...]

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res

def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool_)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res



