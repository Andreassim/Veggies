import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras.utils as image
vid = cv2.VideoCapture(0)
model = load_model("60epoch.h5")

indices = { 0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli', 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'}


while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    
    # If needed, convert the frame to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)'
    if ret:
        input = cv2.resize(frame, dsize= [28, 28])

        test_image = np.expand_dims(input, axis = 0)      

        prediction = model.predict(test_image/255.0)
        index = np.argmax(prediction)

        cv2.putText(frame, f'It is a {indices[index]}', (10,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,(255,255,255), 4, 2)

    # Display the resulting frame
    cv2.imshow('Camera feed', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()