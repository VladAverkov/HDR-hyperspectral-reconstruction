import numpy as np
from base_code.utils import coef_prop


class HsHdr:

    def __init__(self, hsi_l, hsi_m, hsi_h, lower_bound, upper_bound) -> None:
        """
        ldr - low dynamic range image - image with the least exposure (5ms)
        mdr - medium dynamic range image - image with medium exposure (50ms)
        hdr - high dynamic range image - image with the biggest
        exposure (500ms)
        """


        n = 3
        self.cube = hsi_l
        imgs = [hsi_l, hsi_m, hsi_h]
        m = np.zeros((hsi_l.shape[:2]), dtype=bool)
        for i in range(n):
            img = imgs[i]
            m1 = (np.max(img, axis=2) < upper_bound) * (np.min(img, axis=2) > lower_bound)
            m2 = (np.max(self.cube, axis=2) > upper_bound) + (np.min(self.cube, axis=2) < lower_bound)
            mask = m1 * m2 * (1 - m)
            k = coef_prop(img, self.cube, lower_bound, upper_bound)
            for i in range(self.cube.shape[2]):
                self.cube[mask, i] = img[mask, i] * k
            m += m1 * m2
        k = coef_prop(self.cube, hsi_m, lower_bound, upper_bound)
        self.cube *= k


    def get_cube(self):
        return self.cube

    def save_cube(self, path):
        np.save(self.cube, path)

