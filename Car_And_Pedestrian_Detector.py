import cv2

#Pre-trained data
trained_car_data = cv2.CascadeClassifier("cars.xml")
trained_ped_data = cv2.CascadeClassifier("haarcascade_fullbody.xml")

video = cv2.VideoCapture("This is why Pedestrians are more dangerous than vehicles2.mp4")

while True:
    succesful_read, frame = video.read()

    #Covert to gray scale
    gray_scaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Get the cordinates
    car_cord = trained_car_data.detectMultiScale(gray_scaled_image)
    ped_cord = trained_ped_data.detectMultiScale(gray_scaled_image)

    #Draw rectangles
    for (x, y, w, h) in car_cord:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in ped_cord:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Car And Pedestrian Detector", frame)

    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

video.release
