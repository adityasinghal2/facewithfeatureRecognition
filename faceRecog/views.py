from django.shortcuts import render, redirect
import cv2
import numpy as np
from PIL import Image

from time import time
import os
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def index(request):
    return render(request, 'index.html')
def errorImg(request):
    return render(request, 'error.html')

def create_dataset(request):
    
    userId = request.POST['userId']
    print (cv2.__version__)
    #Creating a cascade image classifier
    faceDetect = cv2.CascadeClassifier(BASE_DIR+'/ml/haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    id = userId
 
    sampleNum = 0
    
    while(True):

        ret, img = cam.read()
     
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
       
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)

        for(x,y,w,h) in faces:
            sampleNum = sampleNum+1

            cv2.imwrite(BASE_DIR+'/ml/dataset/user.'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])
          
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
      
            cv2.waitKey(250)

       
        cv2.imshow("Face",img)
       
        cv2.waitKey(1)
        
        if(sampleNum>35):
            break

    cam.release()
  
    cv2.destroyAllWindows()

    return redirect('/')

def trainer(request):

    import os
    from PIL import Image

    #Creating a recognizer to train
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #Path of the samples
    path = BASE_DIR+'/ml/dataset'

    # To get all the images, we need corresponing id
    def getImagesWithID(path):
      
       
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
        #print imagePaths

        
        faces = []
        Ids = []
        for imagePath in imagePaths:
            
            faceImg = Image.open(imagePath).convert('L')

            faceNp = np.array(faceImg, 'uint8')
           
            ID = int(os.path.split(imagePath)[-1].split('.')[1]) # -1 so that it will count from backwards and slipt the second index of the '.' Hence id
            # Images
            faces.append(faceNp)
            # Label
            Ids.append(ID)
            #print ID
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)
        return np.array(Ids), np.array(faces)

    ids, faces = getImagesWithID(path)

   
    recognizer.train(faces, ids)

    # Save the recogzier state so that we can access it later
    recognizer.save(BASE_DIR+'/ml/recognizer/trainingData.yml')
    cv2.destroyAllWindows()

    return redirect('/')


def detect(request):
    faceDetect = cv2.CascadeClassifier(BASE_DIR+'/ml/haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
  
    rec = cv2.face.LBPHFaceRecognizer_create();
    # loading the training data
    rec.read(BASE_DIR+'/ml/recognizer/trainingData.yml')
    getId = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    userId = 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)

            getId,conf = rec.predict(gray[y:y+h, x:x+w]) 

         
            if conf<35:
                userId = getId
                cv2.putText(img, "Detected",(x,y+h), font, 2, (0,255,0),2)
            else:
                cv2.putText(img, "Unknown",(x,y+h), font, 2, (0,0,255),2)

           

        cv2.imshow("Face",img)
        if(cv2.waitKey(1) == ord('q')):
            break
        elif(userId != 0):
            cv2.waitKey(1000)
            cam.release()
            cv2.destroyAllWindows()
            return redirect('/records/details/'+str(userId))

    cam.release()
    cv2.destroyAllWindows()
    return redirect('/')

def detectfeature(request):
 
    

    font = cv2.FONT_HERSHEY_SIMPLEX
    age_net = cv2.dnn.readNetFromCaffe(
        BASE_DIR+'/data/deploy_age.prototxt', 
        BASE_DIR+'/data/age_net.caffemodel')

    gender_net = cv2.dnn.readNetFromCaffe(
        BASE_DIR+'/data/deploy_gender.prototxt', 
        BASE_DIR+'/data/gender_net.caffemodel')
    cap = cv2.VideoCapture(0)

    cap.set(3, 480) #set width
    cap.set(4, 640) #set height

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    gender_list = ['Male', 'Female']



    while True:

        ret, image = cap.read()

        face_cascade = cv2.CascadeClassifier(BASE_DIR+'/data/haarcascade_frontalface_alt.xml')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if(len(faces)>0):
            print("Found {} faces".format(str(len(faces))))

            for (x, y, w, h )in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

                # Get Face 
                face_img = image[y:y+h, h:h+w].copy()
                blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                #Predict Gender
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                print("Gender : " + gender)

                #Predict Age
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                print("Age Range: " + age)
            

                overlay_text = "%s %s" % (gender, age)
                cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow('frame', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       

        return redirect('/')

       






