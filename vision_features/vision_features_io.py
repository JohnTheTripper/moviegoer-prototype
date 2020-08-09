import cv2
import os
import numpy as np


def load_frame(film, frame_number):
    frame_folder = os.path.join('../frame_per_second', film)
    img_path = frame_folder + '/' + film + '_frame' + str(frame_number) + '.jpg'
    frame = cv2.imread(img_path)

    return frame


def unrow_frame(frame):
    unrowed_list = []
    for y in frame:
        for pixel in y:
            unrowed_list.append(pixel)

    unrowed = np.array(unrowed_list)

    return unrowed


def brightest_pixel_intensity(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.flatten()

    return gray.max()


def darkest_pixel_intensity(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.flatten()

    return gray.min()


def remove_highlights_shadows(frame):
    brightest = brightest_pixel_intensity(frame)
    darkest = darkest_pixel_intensity(frame)
    unrowed = unrow_frame(frame)

    if brightest >= 200:

        highlights_removed_list = []
        for pixel in unrowed:
            if pixel.mean() < brightest * .90:
                highlights_removed_list.append(pixel)

        unrowed = np.array(highlights_removed_list)

    if darkest <= 15:
        shadows_removed_list = []
        for pixel in unrowed:
            if pixel.mean() > darkest + 10:
                shadows_removed_list.append(pixel)

        unrowed = np.array(shadows_removed_list)

    return unrowed


def bgr(unrowed):
    b = []
    g = []
    r = []

    for pixel in unrowed:
        b.append(pixel[0])
        g.append(pixel[1])
        r.append(pixel[2])

    b = np.array(b)
    g = np.array(g)
    r = np.array(r)

    return b, g, r


def dominant_color(frame):
    mid_pixels = remove_highlights_shadows(frame)
    b, g, r = bgr(mid_pixels)
    primary_threshold = .5
    secondary_threshold = .1
    if frame.mean() < 30:
        return 0
    elif b.mean() / (mid_pixels.mean() * 3) > primary_threshold:
        return 'blue'
    elif g.mean() / (mid_pixels.mean() * 3) > primary_threshold:
        return 'green'
    elif r.mean() / (mid_pixels.mean() * 3) > primary_threshold:
        return 'red'
    elif b.mean() / (mid_pixels.mean() * 3) < secondary_threshold:
        return 'yellow'
    elif g.mean() / (mid_pixels.mean() * 3) < secondary_threshold:
        return 'magenta'
    elif r.mean() / (mid_pixels.mean() * 3) < secondary_threshold:
        return 'cyan'
    else:
        return 0
