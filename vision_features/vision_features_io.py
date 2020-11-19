import cv2
import os
import numpy as np
import pytesseract


# loading frames

def load_frame(film, frame_number):
    """
    returns a frame as a cv2 image object
    """
    frame_folder = os.path.join('../frame_per_second', film)
    img_path = frame_folder + '/' + film + '_frame_' + str(frame_number) + '.jpg'
    frame = cv2.imread(img_path)

    return frame


def unrow_frame(frame):
    """
    unrows a frame so instead of (x, y, RGB), it's (U, RGB), where U is all of the pixels of the image
    """
    unrowed_list = []
    for y in frame:
        for pixel in y:
            unrowed_list.append(pixel)

    unrowed = np.array(unrowed_list)

    return unrowed


# dimensions

def aspect_ratio(frame):
    """
    returns the aspect ratio of a frame
    """
    height = frame.shape[0]
    width = frame.shape[1]

    return round(width / height, 2)


def blank_frame(frame):
    """
    returns a string if the frame is entirely black or white
    """
    brightest_pixel = max_brightness(frame)
    if frame.mean() < 3:  # threshold of 3, to be safe
        return 'black'
    elif frame.mean() > brightest_pixel * .95:
        return 'white'
    else:
        return None


def center_point(frame):
    """
    returns the (x,y) coordinates of the frame's center point
    """
    height = frame.shape[0]
    width = frame.shape[1]

    half_height = round(height * (1/2))
    half_width = round(width * (1 / 2))
    center_point = (half_width, half_height)

    return center_point


def thirds_points(frame):
    """
    returns the (x,y) coordinates of the frame's four rule-of-thirds points
    """
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
    """
    returns a flag if a row is comprised of entirely black pixels
    useful for determining if frames have artificial aspect ratios
    """
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
    """
    returns the true height of a frame
    useful when frames have artificial aspect ratios
    """
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
    """
    returns a flag if a column is comprised of entirely black pixels
    useful for determining if frames have artificial aspect ratios
    """
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
    """
    returns the true width of a frame
    useful when frames have artificial aspect ratios
    """
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
    """
    returns the true aspect ratio of a frame (if it has an artificial aspect ratio)
    """
    if blank_frame(frame) == 'black' or blank_frame(frame) == 'white':              # ignore blank frames
        return aspect_ratio(frame)
    return round(true_width(frame) / true_height(frame), 2)


# chroma and luma

def mean_brightness(frame):
    """
    returns the mean brightness of a frame
    grayscale conversion is more accurate
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return round(gray.mean())


def calculate_contrast(frame):
    """
    returns the mean brightness of a frame
    grayscale conversion is more accurate
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return round(gray.std())


def max_brightness(frame):
    """
    returns the maximum brightness of a frame
    useful when calculating relative brightness of other pixels
    can be depreciated, duplicate function
    """
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
    """
    returns the maximum brightness of a frame
    useful when calculating relative brightness of other pixels
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.flatten()

    return gray.max()


def darkest_pixel_intensity(frame):
    """
    returns the minimum brightness of a frame
    useful when calculating relative brightness of other pixels
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.flatten()

    return gray.min()


def remove_highlights_shadows(frame):
    """
    returns (unrowed) list of pixels after filtering out any highlights and shadows
    these bright or dark spots may throw off brightness calculations
    """
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
    """
    returns (unrowed) lists of blue, green, and red intensities
    """
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
    """
    returns a dominant color for a frame, if applicable
    may be depreciated and replaced by film_details_io.get_color_shots()
    suffers from long processing time
    """
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
    # check for frames skewed toward a primary color
    if b_mean / (mid_pixels_mean * 3) > primary_threshold:
        return 'blue'
    elif g_mean / (mid_pixels_mean * 3) > primary_threshold:
        return 'green'
    elif r_mean / (mid_pixels_mean * 3) > primary_threshold:
        return 'red'
    # check for frames skewed toward a secondary color (because they lack one of the primary colors)
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
    """
    returns a flag if text is found onscreen
    """
    text = pytesseract.image_to_string(frame)
    if len(text) != 0:
        return True
    else:
        return False
