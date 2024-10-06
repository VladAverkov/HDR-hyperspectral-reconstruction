import numpy as np


class Pair:
    def __init__(self, first, second):
        self.first = first
        self.second = second


def bad_exposure(hs, lower_bound, upper_bound):
    """Returns False if and only if spectra intensity
    in all channels is between lower_bound and upper_bound.
    Otherwise returns True."""
    return (np.max(hs, axis=2) > upper_bound) + (
        np.mean(hs, axis=2) < lower_bound
    )


def good_exposure(hs, lower_bound, upper_bound):
    return (np.max(hs, axis=2) <= upper_bound) * (
        np.mean(hs, axis=2) >= lower_bound
    )


def get_hashbox(rgb, hs, lower_bound, upper_bound, n):

    height, width, _ = rgb.shape

    r_layer, g_layer, b_layer = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    norms = np.linalg.norm(rgb, axis=2)

    phi_del = np.pi / 2 / n
    theta_del = np.pi / 2 / n

    phi = np.zeros((height, width), dtype=float)
    mask1 = r_layer != 0
    phi[mask1] = np.arctan(g_layer[mask1] / r_layer[mask1])
    phi[r_layer == 0] = np.pi / 2
    phi[norms == 0] = 0
    phi = (phi / phi_del).astype(int)

    theta = np.zeros((height, width), dtype=float)
    mask2 = norms != 0
    theta[mask2] = np.arccos(np.clip(b_layer[mask2] / norms[mask2], -1, 1))
    theta[norms == 0] = 0
    theta = (theta / theta_del).astype(int)

    hashbox = theta + phi * (n + 1)
    return hashbox


def reconstruct_hs_hdr(
    rgb, hs, lower_bound, upper_bound, n, sens_matrix
):
     # sens_matrix is a spectral sensitivity matrix used to normalize 
     # hdr_hsi image spectra appropriately. rgb and projected hdr_hsi should be 
     # in the same spectra space. as example you can use xyz color space for both images.
    """Reconstruct spectra in all pixels."""

    hs_hdr_temp = np.copy(hs)

    hashbox = get_hashbox(rgb, hs, lower_bound, upper_bound, n)
    bad_mask = bad_exposure(hs, lower_bound, upper_bound)
    good_mask = good_exposure(hs, lower_bound, upper_bound)

    for hash in np.unique(hashbox):
        curr_bad_mask = (hashbox == hash) * bad_mask
        curr_good_mask = (hashbox == hash) * good_mask
        if np.count_nonzero(curr_good_mask):
            hs_hdr_temp[curr_bad_mask] = np.mean(
                hs[curr_good_mask, :], axis=0
            )

    proj_rgb = np.tensordot(hs_hdr_temp[:, :, :31],
                            sens_matrix, axes=((2), (1)))

    hs_hdr_temp /= np.linalg.norm(proj_rgb, axis=2)[
        ..., np.newaxis
    ]
    hs_hdr_temp *= np.linalg.norm(rgb, axis=2)[
        ..., np.newaxis
    ]

    return hs_hdr_temp
