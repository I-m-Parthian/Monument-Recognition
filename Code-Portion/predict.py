import cv2
import numpy as np
import joblib

img1 = cv2.imread('/Users/parthsmacbookair/Desktop/MR2/Predict/6.JPG')
# load the model from disk
filename = 'Monument-recognition.sav'
model = joblib.load(open(filename,"rb"))
#result = model.score(X_test, Y_test)
print(model.history['acc'])

print(model.history.keys())