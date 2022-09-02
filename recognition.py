import cv2
import imutils
import time
import numpy as np

# {color:[lower,upper]} in hsv
color_dict = {'black':[(43,0, 0),(179,50, 100)],'blue':[(90,47,25),(117,255, 255)],'orange':[(0,119,0),(20,255,255)]}
mask = [0,0,0]
circle_list = [0,0,0]
x = [0,0,0]
y = [0,0,0]
radius = [0,0,0]
center = [0,0,0]


# setting for display font
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontcolor = (0, 0, 255)
fontthickness = 2

vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = vs.read()

    if frame is None:
        break

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    width, height = frame.shape[:2]
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    for i, color in enumerate(color_dict):
        # print(color_dict[color][0], color_dict[color][1])
        mask[i] = cv2.inRange(hsv, color_dict[color][0], color_dict[color][1])
        mask[i] = cv2.erode(mask[i], None, iterations=2)
        mask[i] = cv2.dilate(mask[i], None, iterations=2)
        cv2.imshow(color, mask[i])

        circle_list[i] = cv2.HoughCircles(mask[i].copy(),cv2.HOUGH_GRADIENT,1,20,param1=50,param2=28,minRadius=0,maxRadius=0)
        if type(circle_list[i]) != type(None):

            # return all detected circles
            circles = np.round(circle_list[i][0, :]).astype("int")
            x[i], y[i], radius[i] = circles[0]
            # print( x[i], y[i], radius[i])
            center[i] = (x[i], y[i])
            cv2.circle(frame, center[i], radius[i], (0, 255, 0), 2)
            cv2.circle(frame, center[i], 2, (0,255,0), -1, 8, 0 )
            cv2.putText(frame, color, center[i],font, fontScale, 
                    fontcolor, fontthickness, cv2.LINE_AA, False)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()