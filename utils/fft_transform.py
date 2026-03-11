import numpy as np
import cv2

def fft_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)

    fshift = np.fft.fftshift(f)

    magnitude = 20 * np.log(np.abs(fshift) + 1)

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    magnitude = magnitude.astype("float32")

    return magnitude