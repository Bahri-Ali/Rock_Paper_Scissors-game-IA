import cv2 as cv
import mediapipe as mp



video = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
lms = []
while True :
    ret , fram = video.read()
    fram = cv.flip(fram,1)
    imgRGB = cv.cvtColor(fram , cv.COLOR_BGR2RGB)
    # hand detect  
    result  = hands.process(imgRGB)
    if result.multi_hand_landmarks : 
    #     print('I show a Hand')
        for handlms in result.multi_hand_landmarks :
           for id , lm in enumerate(handlms.landmark):
                h ,w , c = fram.shape
                cx,cy = int(lm.x * w) , int(lm.y * h)
                lms.append([id , cx ,cy])
                mpDraw.draw_landmarks(fram , handlms , mpHands.HAND_CONNECTIONS)
                if id == 8 :
                   cv.circle(fram , (cx , cy) ,10 ,(0,200,0),cv.FILLED)
                if id == 6:
                    cv.circle(fram , (cx , cy) , 10 ,(0 , 200 , 0) , cv.FILLED)
                if len(lms) >= 21:
                    if lms[8][2] < lms[6][2]:
                        print('up')
                    else:
                        print('down')
    cv.imshow('pcauto'  , fram)
    if cv.waitKey(1) == ord(' '):
       break
video.release()  
cv.destroyAllWindows()