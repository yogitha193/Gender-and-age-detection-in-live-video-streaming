from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy
import numpy as np
import cv2
import os
import cvlib as cv
 
#load model
model= load_model('Age_Gender_Detection.h5')
print("Model loaded ")

#open webcam
webcam= cv2.VideoCapture(0)
print("Webcam loaded")



while webcam.isOpened():

    #read frame from webcam
    status, frame = webcam.read()

    #apply face detection
    face, confidence = cv.detect_face(frame)

    #loop through detected faces
    for idx, f in enumerate(face):

        #get corner points of face rectangle
        (startx, starty) = f[0],f[1]
        (endx, endy)= f[2],f[3]

        #draw rectangle over face
        cv2.rectangle(frame, (startx,starty), (endx,endy), (0,255,0), 2)

        #crop the detected face region
        face_crop = np.copy(frame[starty:endy,startx:endx])

        if(face_crop.shape[0]) <10 or (face_crop.shape[1]) <10:
            continue

        print(face_crop.shape)    
        #preprocessing for gender and face detection model
        face_crop = cv2.resize(face_crop, (48,48,))
        face_crop=numpy.expand_dims(face_crop,axis=0)
        face_crop=np.array(face_crop)
        face_crop=np.delete(face_crop,0,1)
        face_crop=np.resize(face_crop,(48,48,3))
        print(face_crop.shape)
        sex_f=("Male","Female")

        face_crop=np.array([face_crop])/255

        #face_crop = face_crop.astype("float")/255.0
        #face_crop = img_to_array(face_crop)
        #face_crop = np.expand_dims(face_crop, axis=0)

        #apply gender and age detection on face
        pred = model.predict(face_crop)
        #print(pred)
        age=int(np.round(pred[1][0]))
        sex=int(np.round(pred[0][0]))

        print("Predicted age is "+str(age))
        #print(sex)
        print("Predicted Gender is "+sex_f[sex])

        label = sex_f[sex]
        label = "{}: {}".format(label,age)
        y= starty-10 if starty-10> 10 else starty+10

        cv2.putText(frame, label, (startx, y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.imshow("gender detection",frame)

    if cv2.waitKey(200) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()
