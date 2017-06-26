#!/usr/bin/env python
""" Starting script for the face recognition system.
"""

import sys
import os
import numpy as np
from face_recognition_system.videocamera import VideoCamera
from face_recognition_system.detectors import FaceDetector
import face_recognition_system.operations as op
import cv2
from cv2 import __version__

from matplotlib.pyplot import imshow
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = (12, 8)
import operator



def get_images(frame, faces_coord, shape):
    """ Perfrom transformation on original and face images.

    This function draws the countour around the found face given by faces_coord
    and also cuts the face from the original image. Returns both images.

    :param frame: original image
    :param faces_coord: coordenates of a rectangle around a found face
    :param shape: indication of which shape should be drwan around the face
    :type frame: numpy array
    :type faces_coord: list of touples containing each face information
    :type shape: String
    :return: two images containing the original plus the drawn contour and
             anoter one with only the face.
    :rtype: a tuple of numpy arrays.
    """
    if shape == "rectangle":
        faces_img = op.cut_face_rectangle(frame, faces_coord)
        frame = op.draw_face_rectangle(frame, faces_coord)
    elif shape == "ellipse":
        faces_img = op.cut_face_ellipse(frame, faces_coord)
        frame = op.draw_face_ellipse(frame, faces_coord)
    faces_img = op.normalize_intensity(faces_img)
    faces_img = op.resize(faces_img)
    return (frame, faces_img)

def add_person(people_folder, shape):
    """ Funtion to add pictures of a person

    :param people_folder: relative path to save the person's pictures in
    :param shape: Shape to cut the faces on the captured images:
                  "rectangle" or "ellipse"
    :type people_folder: String
    :type shape: String
    """
    person_name = raw_input('What is the name of the new person: ').lower()
    folder = people_folder + person_name
    if not os.path.exists(folder):
        raw_input("I will now take 20 pictures. Press ENTER when ready.")
        os.mkdir(folder)
        video = VideoCamera()
        detector = FaceDetector('face_recognition_system/frontal_face.xml')
        counter = 1
        timer = 0
        cv2.namedWindow('Video Feed', cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)
        mood_adj = "How do you look like?"
        while counter < 21:
            frame = video.get_frame()
            face_coord = detector.detect(frame)
            if len(face_coord):
                cv2.putText(frame, mood_adj.capitalize(),
                            (face_coord[0][0], face_coord[0][1]),
                            cv2.FONT_HERSHEY_PLAIN, 1.7, (206, 0, 209), 2,
                            cv2.LINE_AA)

                frame, face_img = get_images(frame, face_coord, shape)
                # save a face every second (100), we start from an offset '5' because
                # the first frame of the camera gets very high intensity
                # readings.
                if timer % 100 == 5:
                    
                    (conf_adj, mood_adj) = analyze_face(frame) ### TEST
                    cv2.imwrite(folder + '/' + str(counter) + '.jpg',
                                face_img[0])
                    print 'Images Saved:' + str(counter)
                    counter += 1
                    #cv2.imshow('Saved Face', face_img[0])

            cv2.imshow('Video Feed', frame)
            cv2.waitKey(50)
            timer += 5
    else:
        print "This name already exists."
        sys.exit()

def analyze_face(img):
    import requests
    import PIL
    import StringIO
    f = StringIO.StringIO()
    PIL.Image.fromarray(img).save(f, 'png')
    data = f.getvalue()
    print "OK"
    from donthackme import API_KEY
    endpoint = 'https://westeurope.api.cognitive.microsoft.com/face/v1.0/detect'
    args = {'returnFaceId': 'true',
            'returnFaceLandmarks': 'false',
            'returnFaceAttributes': 'age,gender,emotion,smile,glasses'}
    headers = {'Content-Type': 'application/octet-stream',
               'Ocp-Apim-Subscription-Key': API_KEY}
    response = requests.post(data=data,url=endpoint,headers=headers,params=args)
    conf_adj= ""
    mood_adj = ""
    for face in response.json():
        print face
        # sadness, neutral, contempt, disgust, anger, surprise, fear, happiness
        infos = face['faceAttributes']
        mood = infos['emotion']
        age = infos['age']
        gender = infos['gender']
        glasses = infos['glasses']
        smile = infos['smile']
        if gender == 'male':
            print "\nYet another boy!"
        agish = int(round5(age))
        print "You seem close to {agish}!!  Are you {age} years old maybe?!".format(agish=agish, age=age)

        sorted_mood = sorted(mood.items(), key=operator.itemgetter(1))
        mood, conf = sorted_mood[-1]
        mood_adj = {'sadness':'sad', 'neutral':'ok', 'contempt':'contempt', 'disgust':'disgust',
                    'anger':'angry', 'surprise':'surprised', 'fear':'afraid', 'happiness':'happy'}[mood]
        if conf > 0.9:
            conf_adj = 'very'
        elif conf > 0.5:
            conf_adj = 'quite'
        elif conf > 0.3:
            conf_adj = 'maybe'
        elif conf > 0.1:
            conf_adj = 'slightly'
        else:
            conf_adj = 'barely'
            print "You look {conf_adj} {mood}".format(conf_adj=conf_adj, mood=mood_adj)
        if glasses == "reading glasses":
            print "Cool glasses by the way!"
        if smile > 0.7:
            print "Yeaaahhh! That's a BIG SMILE!"
    return (conf_adj, mood_adj)

        
def round5(x, base=5):
    return int(base * round(float(x)/base))

def recognize_people(people_folder, shape):
    """ Start recognizing people in a live stream with your webcam

    :param people_folder: relative path to save the person's pictures in
    :param shape: Shape to cut the faces on the captured images:
                  "rectangle" or "ellipse"
    :type people_folder: String
    :type shape: String
    """
    try:
        people = [person for person in os.listdir(people_folder)]
    except:
        print "Have you added at least one person to the system?"
        sys.exit()
    print "This are the people in the Recognition System:"
    for person in people:
        print "-" + person

    print 30 * '-'
    print "   POSSIBLE RECOGNIZERS TO USE"
    print 30 * '-'
    print "1. EigenFaces"
    print "2. FisherFaces"
    print "3. LBPHFaces"
    print 30 * '-'

    choice = check_choice()

    detector = FaceDetector('face_recognition_system/frontal_face.xml')
    if choice == 1:
        recognizer = cv2.face.createEigenFaceRecognizer()
        threshold = 4000
    elif choice == 2:
        recognizer = cv2.face.createFisherFaceRecognizer()
        threshold = 300
    elif choice == 3:
        recognizer = cv2.face.createLBPHFaceRecognizer()
        threshold = 85 #105
    images = []
    labels = []
    labels_people = {}
    for i, person in enumerate(people):
        labels_people[i] = person
        for image in os.listdir(people_folder + person):
            images.append(cv2.imread(people_folder + person + '/' + image, 0))
            labels.append(i)
    try:
        recognizer.train(images, np.array(labels))
    except:
        print "\nOpenCV Error: Do you have at least two people in the database?\n"
        sys.exit()

    video = VideoCamera()
    while True:
        frame = video.get_frame()
        faces_coord = detector.detect(frame, False)
        if len(faces_coord):
            frame, faces_img = get_images(frame, faces_coord, shape)
            for i, face_img in enumerate(faces_img):
                if __version__ == "3.1.0":
                    collector = cv2.face.MinDistancePredictCollector()
                    recognizer.predict(face_img, collector)
                    conf = collector.getDist()
                    pred = collector.getLabel()
                else:
                    pred, conf = recognizer.predict(face_img)
                print "Prediction: " + str(pred)
                print 'Confidence: ' + str(round(conf))
                print 'Threshold: ' + str(threshold)
                if conf < threshold:
                    cv2.putText(frame, labels_people[pred].capitalize(),
                                (faces_coord[i][0], faces_coord[i][1] - 2),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (206, 0, 209), 2,
                                cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Unknown",
                                (faces_coord[i][0], faces_coord[i][1]),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (206, 0, 209), 2,
                                cv2.LINE_AA)

        # cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
        #             cv2.FONT_HERSHEY_PLAIN, 1.2, (206, 0, 209), 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)
        if cv2.waitKey(100) & 0xFF == 27:
            sys.exit()

def check_choice():
    """ Check if choice is good
    """
    is_valid = 0
    while not is_valid:
        try:
            choice = int(raw_input('Enter your choice [1-3] : '))
            if choice in [1, 2, 3]:
                is_valid = 1
            else:
                print "'%d' is not an option.\n" % choice
        except ValueError, error:
            print "%s is not an option.\n" % str(error).split(": ")[1]
    return choice

if __name__ == '__main__':
    print 30 * '-'
    print "   POSSIBLE ACTIONS"
    print 30 * '-'
    print "1. Add person to the recognizer system"
    print "2. Start recognizer"
    print "3. Exit"
    print 30 * '-'

    CHOICE = check_choice()

    PEOPLE_FOLDER = "face_recognition_system/people/"
    SHAPE = "ellipse"

    if CHOICE == 1:
        if not os.path.exists(PEOPLE_FOLDER):
            os.makedirs(PEOPLE_FOLDER)
        add_person(PEOPLE_FOLDER, SHAPE)
    elif CHOICE == 2:
        recognize_people(PEOPLE_FOLDER, SHAPE)
    elif CHOICE == 3:
        sys.exit()

