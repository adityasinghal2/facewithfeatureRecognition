# facial-recognition-python-django
Face detection and facial recognition along with recognized persons information fetched from database.


Setup:
run:
pip install -r requirements.txt # to install all the packages



General Languages and versions

Django                2.1.4
opencv-contrib-python 3.4.4.19
Pillow                5.3.0

git clone https://github.com/adityasinghal2/facewithfeatureRecognition.git
cd facewithfeatureRecognition

Run -
    python manage.py migrate # to migrate madels
    python manage.py runserver # to start the server
Open http://localhost:8000/ in browser
    
Features:
1:create Dataset:
        
        
        
        i.click on it it will open webcam then it will start detecting the face and storing in dataset folder till 30sec and closes.
       
       
       ii. fill the details of the particular image id(dataset)
            >> python manage.py createsuperuser
            >> When prompted, type your username (lowercase, no spaces), email address, and password
            >> Return to your browser. Log in with the superuser's credentials you chose; you should see the Django admin dashboard
            >> fill the detail of particular dataset id by clicking add option.


2: Train the classifier
        click on it to train the classifier with stored dataset.



3:Detect the face with webcam
        click on it to detect the face against trained dataset
 
 
 
 
 
 4:Detect age and gender By Webcam
        click on it to detect age and gender
 
 


    

