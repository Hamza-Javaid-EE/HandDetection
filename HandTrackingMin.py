import cv2
import mediapipe as mp
import time

#Camera Access
cap = cv2.VideoCapture(0)

#Using hand Detection Module Builtin
mpHands = mp.solutions.hands
hands = mpHands.Hands() #object of hand
mpDraw = mp.solutions.drawing_utils

cTime=0
pTime=0

#Calling Camera
while True:
    success, img = cap.read()

    #Image to RGB
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)   #Process the frame of hands

    #Extract the multiple Hands
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            #Access each 20 points separately
            for id,lm in enumerate(handlms.landmark):
                #This gives the each point(id) and there landmark(x,y,z) location
                #x,y,z is ratio so we convert into the pixel bt multiplying it with width and height
                h,w,c = img.shape #C for coloumn
                cx, cy = int(lm.x * w) , int(lm.y * h) #Centre point

                #Controling each id(point)/ Tip of Fingers
                # if id==4:
                #     cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
                # if id==8:
                #     cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
                # if id==12:
                #     cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
                # if id==16:
                #     cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
                # if id==20:
                #     cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            #Drawing the 21 points with the mediaPipe
            # handlms is hand lanmarks for each hand
            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)
    # FPS Calculation
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)


    cv2.imshow("Image",img)
    cv2.waitKey(1)