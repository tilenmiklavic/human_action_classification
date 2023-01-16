import cv2
import numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
 
def correct_rotation(frame, rotate_code):  
    return cv2.rotate(frame, rotate_code) 

# Read the video from file
video = cv2.VideoCapture("input.MOV")
fps = video.get(cv2.CAP_PROP_FPS) 
video_fps = video.get(cv2.CAP_PROP_FPS),
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)

fourcc = cv2.VideoWriter_fourcc(*'X264')
writer = cv2.VideoWriter('OUTPUT_PATH.mp4', apiPreference=0, fourcc=fourcc, fps=video_fps[0], frameSize=(int(width), int(height)))

rotateCode = cv2.ROTATE_180

out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))



# Check if the video was opened successfully
if not video.isOpened():
    print("Error opening the video file.")
    exit()

# Read the first frame of the video
_, frame = video.read()

while(True):
    # Capture frame-by-frame
    ret, frame = video.read()
    if rotateCode is not None:
         frame = correct_rotation(frame, rotateCode)
         
    # resizing for faster detection
    # frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    # cv2.imshow('frame',frame)
    
    frame.shape[0]
    
    print(frame.shape[0])
    
    # initialize our video writer
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    # writer = cv2.VideoWriter('output1.mp4', fourcc, fps, (int(frame.shape[0]), int(frame.shape[1])))
            
    writer.write(frame)
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# and release the output
out.release()
writer.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)