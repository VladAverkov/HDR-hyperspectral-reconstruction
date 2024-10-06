from base_code.reconstructor import reconstruct_hs_hdr


def get_hdr_hsi(lower_bound=0, upper_bound=0.97):
    # hdr_rgb = your_hdr_rgb_image # hdr_rgb.shape = (H, W, 3)
    # hsi = your_hsi_hdr_image # hsi.shape = (H, W, n_channels)
    # hdr_hsi = reconstruct_hs_hdr(hdr_rgb, hsi, lower_bound, upper_bound, n=200)
    return 0


if __name__ == "__main__":
    get_hdr_hsi()
