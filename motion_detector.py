import cv2, time

first_frame = None

#   0 è l'indice della webcam
video = cv2.VideoCapture(0)


#   check è un boolean datatype (true or false) , frame è un numpy array ( [121...1515])
while True :
    check, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #   Aggiungo una sfocatura perchè riduce il rumore dell'immagine ed aumenta l'accuratezza dell'algoritmo
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None :
        first_frame = gray
        continue

    #   Differenza tra il primo frame e il frame corrente
    delta_frame = cv2.absdiff(first_frame, gray)

    #   30 è il limite di thresh che assegno, 255 è il colore bianco che assegno all'immagine in movimento | [1] -> voglio l'accesso al secondo item della tupla
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    #   Definizione dei contorni
    (_,cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #   Creo il rettangolo
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)

    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows
