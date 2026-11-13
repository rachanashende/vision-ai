
import sys
import tensorflow as tf
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import sklearn
import matplotlib

print("========== ASL PROJECT SETUP ==========")

print("Python:", sys.version.split()[0])
print("TensorFlow:", tf.__version__)
print("MediaPipe:", mp.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)

print("=======================================")
print("SUCCESS! All required libraries imported.")
