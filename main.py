import os
import cv2 as cv
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox
import pandas as pd
import datetime, time

root = Tk()
root.title('-:Face Recognization Model:-')

image1 = Image.open(r"C:\Users\Mr-Singh\Downloads\fdp.jpg")
test = ImageTk.PhotoImage(image1)
label1 = Label(image=test)
label1.place(x=0, y=-50)

title = Label(root, text='~~ Face Recognization Model ~~', bg='lightsteelblue2')
title.config(font=('helvetica', 18))
title.place(x=170, y=5)

lblid = Label(root, text='Enter Roll No:', font='ariel 12')
lblid.config(font=('helvetica', 12, 'bold'), bg='lightblue')
lblid.place(x=130, y=441)
ent0= Entry(root,font=10, width=30, highlightthickness=2)
ent0.config(highlightbackground = "red", highlightcolor= "red")
ent0.place(x=280, y=440)

lblc = Label(root, text='Enter your name:', font='ariel 12')
lblc.config(font=('helvetica', 12, 'bold'), bg='lightblue')
lblc.place(x=130, y=481)
ent1= Entry(root,font=10, width=30, highlightthickness=2)
ent1.config(highlightbackground = "red", highlightcolor= "red")
ent1.place(x=280, y=480)

path = 'Tests/Face Data/'

def capface():
    name = (ent1.get())
    entid = (ent0.get())
    if (entid == ''):
        msgbox = messagebox.showinfo("Entry Empty: ", "Kindly Fill your r.no to start.. ")
        return msgbox
    elif (name == ''):
        msgbox = messagebox.showinfo("Entry Empty: ", "Kindly Fill your name to start.. ")
        return msgbox
    else:
        if not os.path.exists(entid):
            folder = os.path.join(path, entid)
            try:
                os.makedirs(folder)
            except OSError as error:
                messagebox.showinfo("Exist..!", "Roll number " + entid + " alreaddy Exists..")

            else:
                folder1 = os.path.join(folder, name)
                os.mkdir(folder1)
                print("Directory ", entid + " and name ", name + " Created ")
                messagebox.showinfo("Created..",
                                    "Roll number: " + entid + " and name: " + name + " Added Sucessfully..")
                ent1.delete(0, END)
                ent1.insert(0, 'Roll no. and name added sucessfully')

                cam = cv.VideoCapture(0)
                harcascadePath = "haar_face.xml"
                detector = cv.CascadeClassifier(harcascadePath)
                sample = 0
                while (True):
                    ret, img = cam.read()
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv.rectangle(img, (h,w-35), (y,w), (0,255,0), cv.FILLED)
                        cv.putText(img, 'Taking Samples..', (h+6, w-6), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
                            # incrementing sample number
                        sample = sample + 1
                            # saving the captured face in the dataset folder TrainingImage
                        cv.imwrite("{}\ ".format(folder1) + name + "_" + str(sample) + ".jpg", gray[y:y + h, x:x + w])
                            # display the frameC:\Users\Mr-Singh\Documents\Python Programming\os file directory practice
                        cv.imshow('frame', img)
                            # wait for 100 miliseconds
                    if cv.waitKey(100) & 0xFF == ord('q'):
                        break
                            # break if the sample number is morethan 50
                    elif sample > 50:
                        MsgBox = messagebox.showinfo("sample-info", "Sample Has Taken Sucessfully..")
                        if MsgBox == 'ok':
                            break
                cam.release()
                cv.destroyAllWindows()

capturebtn=Button(root, text = 'Capture Face', command=capface, bg='brown', fg='white', font=('helvetica', 12, 'bold'))
capturebtn.place(x=260, y=550)



def trainface():
    people = []
    for i in os.listdir("Tests/Face Data"):
        for sub in os.listdir("Tests/Face Data/" + i):
            print('sub is : ' + sub)
            people.append(sub)

    features = []
    labels = []
    # dir = os.listdir("Tests/Face Data")
    haar_cascade = cv.CascadeClassifier(r'C:\Users\Mr-Singh\PycharmProjects\FR-Model BCA\haar_face.xml')

    def create_train():
        for a in os.listdir("Tests/Face Data"):
            rno_path = ("Tests/Face Data/" + a)
            print("rno path: " + rno_path)
            print('Roll No: ' + a)
            for stname in os.listdir(rno_path):
                print('st name: ' + stname)
                stimgpath = os.path.join(rno_path, stname)
                label = people.index(stname)

                for img in os.listdir(stimgpath):
                    print('img path: ' + img)
                    img_path = os.path.join(stimgpath, img)

                    img_array = cv.imread(img_path)
                    gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

                    faces_rect = haar_cascade.detectMultiScale(img_array, scaleFactor=1.1, minNeighbors=4)

                    for (x, y, w, h) in faces_rect:
                        faces_roi = gray[y:y + h, x:x + w]
                        features.append(faces_roi)
                        labels.append(label)

    create_train()

    features = np.array(features, dtype='object')
    labels = np.array(labels)

    print("Length of the fearures {}".format(len(features)))
    print("Length of the Labels {}".format(len(labels)))

    # Train the Recognizer on the features list and the labels list

    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    face_recognizer.train(features, labels)
    # Save the trained file in np array
    face_recognizer.save("face_trained.yml")
    np.save("features.npy", features)
    np.save("laebls.npy", labels)
    messagebox.showinfo("Completed", "System has Trained and added the Face Sucessfully.. ;-)")


trainbtn=Button(root, text = 'Train your Face', command=trainface, bg='blue', fg='white', font=('helvetica', 12, 'bold'))
trainbtn.place(x=390, y=550)




def start():
    # getting the name from "userdetails.csv"
    # attendance = pd.read_csv("attendance.csv")
    colms = ['Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns = colms)
    peopl = []
    for i in os.listdir("Tests/Face Data"):
        print('Roll no: ' +i)
        for sub in os.listdir("Tests/Face Data/" + i):
            print('Name is : ' + sub)
            peopl.append(sub)

    features = np.load("features.npy", allow_pickle=True)
    labels = np.load("laebls.npy")
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read("face_trained.yml")
    vid = cv.VideoCapture(0)

    while (True):
        # Capture the video frame
        # by frame
        isTrue, img = vid.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # import face recognisation file
        haar_cascade = cv.CascadeClassifier('haar_face.xml')
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
        print(f'Number of faces found = {len(faces_rect)}')

        for (y1, x2, y2, x1) in faces_rect:
            faces_roi = gray[x2:x2 + y2, y1:y1 + x1]
            lab, conf = face_recognizer.predict(faces_roi)
            # print("Label = {} with a confidence of {}".format(people[label], confidence))
            # Put Name on the detected face
            # If confidence is less them 100 ==> "0" : perfect match

            if (conf < 90):
                if peopl[lab] in peopl:
                    tt=str(peopl[lab])
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                    attendance.loc[len(attendance)] = [str(tt), date, timestamp]
                    co=0,255,0
            else:
                tt = str('unknown')
            if (conf > 90):
                noOfFile = len(os.listdir("Tests/Unknown")) + 1
                cv.imwrite("Tests/Unknown/Image " + str(noOfFile) + ".jpg", faces_roi)
                co = 0, 0, 255
            x1,y1,x2,y2 = x1,y1,x2*2,y2 + x2-4
            cv.rectangle(img, (x1, y1), (x2, y2), (co), thickness=1)
            cv.rectangle(img, (x1,y2-35,), (x2,y2), (0, 255, 0), cv.FILLED)
            cv.putText(img, str(tt), (x1-6, y2+6), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))
            # cv.putText(img, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(keep='first', subset=['Name'])
        cv.imshow('Recognizing Faces... ', img)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    # img= img.release()
    # cv.imshow("Face", img.release())

    # Destroy all the windows
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timestamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    vid.release()

    cv.destroyAllWindows()
    print(attendance)
    res=attendance
    messagebox.showinfo("Completed", "Attendence has been Taken Sucessfully.. ;-)\n {} ".format(res))



startbtn=Button(root, text = ' Recognize Face ', command=start, bg='green', fg='white', font=('helvetica', 12, 'bold'))
startbtn.place(x=535, y=550)


def clearscr():
    ent0.delete(0,END)
    ent1.delete(0, END)



clearoutput= Button(root, text=' Clear Screen ', command=clearscr, bg='purple', fg='white',
              font=('helvetica', 12, 'bold'))
clearoutput.place(x=470, y=600)

def exit():
    MsgBox = messagebox.askquestion('Exit Application', 'Are you sure, you want to exit the application ?',
                                    icon='warning')
    if MsgBox == 'yes':
        root.destroy()

exit = Button(root, text='  Exit  ', command=exit, bg='red', fg='white',
              font=('helvetica', 12, 'bold'))
exit.place(x=620, y=600)


root.geometry('700x650')
root.configure(background='white')
root.mainloop()
