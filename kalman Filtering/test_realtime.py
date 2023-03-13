from frame_processing import frame_process


import cv2
  
def main(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():

        _, frame = cap.read()
        if frame is None:
            break

        cv2.imshow('', frame_process(frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main("C:\\Users\\shree\\Downloads\\IMG_0264.MOV")
