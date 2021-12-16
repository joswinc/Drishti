import face_recognition
import cv2
import numpy as np
import os, os.path
from os import listdir
from os.path import isfile, join

face_update = True
    


def check_known_face():
    DIR = 'known_faces'
    #count_faces = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    
    known_images = [f for f in listdir(DIR) if isfile(join(DIR, f))]
    known_face_names = []
    known_face_encodings = []
    for i in known_images:
        known_face_names.append(i[:-4])
        known_image = face_recognition.load_image_file(DIR+'/'+i)
        face_locations = face_recognition.face_locations(known_image)
        known_face_encoding = face_recognition.face_encodings(known_image, face_locations)[0]
        known_face_encodings.append(known_face_encoding)
    
    return known_face_encodings,known_face_names

def face_recog():
    
    global face_update
    if face_update==True:
        known_face_encodings,known_face_names = check_known_face()
        face_update=False
        
    video_capture = cv2.VideoCapture(0)
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
    
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
    
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
    
                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]
    
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
    
                face_names.append(name)
    
        process_this_frame = not process_this_frame
    
        return face_names
        
    
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
def face_store(name):
    #name = '123'
    global face_update
    
    video_capture = cv2.VideoCapture(0)
    process_this_frame = True
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
    
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
    
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
         
        process_this_frame = not process_this_frame

        if len(face_encodings)==1:
            exists = os.path.isfile('known_faces/'+name+'.jpg')
            if exists:
                return ("File Already Exists")
                break
            else:
                face_update=True
                cv2.imwrite('known_faces/'+name+'.jpg',frame)
                return ("Face saved")
                break
        elif len(face_encodings)==0:
            return ("No face found")
        else:
            return ("Multiple faces found")
        
    
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

