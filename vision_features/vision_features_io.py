import cv2
import os
import numpy as np
import pytesseract
import face_recognition

# loading frames


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


# dimensions

def aspect_ratio(frame):
    height = frame.shape[0]
    width = frame.shape[1]

    return round(width / height, 2)


def blank_frame(frame):
    brightest_pixel = max_brightness(frame)
    if frame.mean() < 3:  # threshold of 3, to be safe
        return 'black'
    elif frame.mean() > brightest_pixel * .95:
        return 'white'
    else:
        return None


def center_point(frame):
    height = frame.shape[0]
    width = frame.shape[1]

    half_height = round(height * (1/2))
    half_width = round(width * (1 / 2))
    center_point = (half_width, half_height)

    return center_point


def thirds_points(frame):
    height = frame.shape[0]
    width = frame.shape[1]

    one_third_height = round(height * (1 / 3))
    two_thirds_height = round(height * (2 / 3))
    one_third_width = round(width * (1 / 3))
    two_thirds_width = round(width * (2 / 3))

    thirds_point_a = (one_third_width, one_third_height)            # upper-left
    thirds_point_b = (two_thirds_width, one_third_height)           # upper-right
    thirds_point_c = (one_third_width, two_thirds_height)           # bottom-left
    thirds_point_d = (two_thirds_width, two_thirds_height)          # bottom-right

    return thirds_point_a, thirds_point_b, thirds_point_c, thirds_point_d


def black_row(frame, row_selection):
    total_pixels = frame.shape[1]
    black_pixels = 0

    for pixel in frame[row_selection]:
        if pixel.mean() < 5:
            black_pixels += 1

    if black_pixels == total_pixels:
        return True
    else:
        return False


def true_height(frame):
    top_row = 0
    bottom_row = frame.shape[0] - 1
    search_flag = True
    first_flag = True

    while search_flag == True:
        if black_row(frame, top_row) and black_row(frame, bottom_row):
            top_row += 1
            bottom_row -= 1
            first_flag = False
        elif first_flag == True:
            return frame.shape[0]  # if the first search doesn't yield black columns, simply return frame height
        else:
            search_flag = False

        bottom_row += 1  # necessary because we had to subtract 1 when declaring bottom_row

    return bottom_row - top_row


def black_column(frame, col_selection):
    total_pixels = frame.shape[0]
    black_pixels = 0

    for row in frame:
        if row[col_selection].mean() < 5:
            black_pixels += 1

    if black_pixels == total_pixels:
        return True
    else:
        return False


def true_width(frame):
    left_column = 0
    right_column = frame.shape[1] - 1  # list starts at 0, while .shape starts at 1
    search_flag = True
    first_flag = True

    while search_flag == True:
        if black_column(frame, left_column) and black_column(frame, right_column):
            left_column += 1
            right_column -= 1
            first_flag = False
        elif first_flag == True:  # if the first search doesn't yield black columns, simply return frame width
            return frame.shape[1]
        else:
            search_flag = False

        right_column += 1  # necessary because we had to subtract 1 when declaring right_column

    return right_column - left_column


def true_aspect_ratio(frame):
    if blank_frame(frame) == 'black' or blank_frame(frame) == 'white':
        return aspect_ratio(frame)
    return round(true_width(frame) / true_height(frame), 2)


# chroma and luma

def mean_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return round(gray.mean(), 2)


def calculate_contrast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return round(gray.std(), 2)


def max_brightness(frame):
    brightest_pixel = 0
    # brightest_coordinate = 0
    x = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for a in gray:
        y = 0
        for b in a:
            if b > brightest_pixel:
                brightest_pixel = b
                # brightest_coordinate = (x, y)
            y += 1
        x += 1
    return brightest_pixel


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
    if frame.mean() < 50:
        return None

    # resize for faster processing
    scale_percent = 30  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    mid_pixels = remove_highlights_shadows(frame)
    mid_pixels_mean = mid_pixels.mean()
    b, g, r = bgr(mid_pixels)
    b_mean = b.mean()
    g_mean = g.mean()
    r_mean = r.mean()
    primary_threshold = .5
    secondary_threshold = .1
    if b_mean / (mid_pixels_mean * 3) > primary_threshold:
        return 'blue'
    elif g_mean / (mid_pixels_mean * 3) > primary_threshold:
        return 'green'
    elif r_mean / (mid_pixels_mean * 3) > primary_threshold:
        return 'red'
    elif b_mean / (mid_pixels_mean * 3) < secondary_threshold:
        return 'yellow'
    elif g_mean / (mid_pixels_mean * 3) < secondary_threshold:
        return 'magenta'
    elif r_mean / (mid_pixels_mean * 3) < secondary_threshold:
        return 'cyan'
    else:
        return None


# onscreen text

def onscreen_text(frame):
    text = pytesseract.image_to_string(frame)
    if len(text) != 0:
        return True
    else:
        return False



