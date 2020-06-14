import cv2
import os

# create frames from video file
title = 'extremely_wicked'
scale_percent = 100     # optional image scaling

cap = cv2.VideoCapture('input_videos/' + title + '.mkv')
output_directory = os.path.join('dialogue_frames/', title)
os.mkdir(output_directory)

if not cap.isOpened():
    print("Error opening video stream or file")

frame_rate = cap.get(5)              # frame rate
print('Video frame rate:', frame_rate)
frame_rate_int = round(frame_rate)             # rounds up the frame_rate of 23.976 to 24 for cleaner frame extraction
print('Extracting image every:', frame_rate_int, 'frames')

count = 1   # starts at 1, because first image isn't saved until 24th frame (timestamp of 1 second)
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    if cap.get(1) % frame_rate_int == 0:     # frame every 1 second
        filename = output_directory + '/' + title + '_frame%d.jpg' % count

        # optional image scaling
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dimensions = (width, height)
        frame = cv2.resize(frame, dimensions)

        print(cap.get(0), cap.get(1), count)
        cv2.imwrite(filename, frame)
        count += 1

cap.release()
print('Extracted', count, 'frames')
