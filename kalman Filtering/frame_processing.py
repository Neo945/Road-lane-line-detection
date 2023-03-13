import cv2
import numpy as np
from direction import direction_processing

import track
import detect


lt = track.LaneTracker(2, 0.1, 500)
ld = detect.LaneDetector(180)
    
ticks = 0

def frame_process(frame):

    global ticks
    precTick = ticks
    ticks = cv2.getTickCount()
    dt = (ticks - precTick) / cv2.getTickFrequency()

    if frame is None:
        return None

    predicted = lt.predict(dt)

    lanes = ld.detect(frame)

    if predicted is not None:
        cv2.line(frame,
                    np.array((predicted[0][0][0], predicted[0][1][0]), dtype=np.int32),
                    np.array((predicted[0][2][0], predicted[0][3][0]), dtype=np.int32),
                    (0, 0, 255), 5)
        cv2.line(frame,
                    np.array((predicted[1][0][0], predicted[1][1][0]), dtype=np.int32),
                    np.array((predicted[1][2][0], predicted[1][3][0]), dtype=np.int32),
                    (0, 0, 255), 5)

    if lanes is not None:
        lt.update(lanes)
    # return frame
    return direction_processing(frame)

    # frame = frame_processing(frame)
    # cv2.imshow('', frame)


