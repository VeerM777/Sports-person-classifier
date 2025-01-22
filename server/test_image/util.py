import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import os

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(image_base64_data, file_path=None):
    # Ensure image is processed correctly
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1, len_image_array).astype(float)

        # Predict the class and class probability
        try:
            prediction = __model.predict(final)
            class_name = class_number_to_name(prediction[0])
            class_prob = np.around(__model.predict_proba(final) * 100, 2).tolist()[0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            class_name = "Unknown"
            class_prob = []

        result.append({
            'class': class_name,
            'class_probability': class_prob,
            'class_dictionary': __class_name_to_number
        })

    return result

def class_number_to_name(class_num):
    return __class_number_to_name.get(class_num, "Unknown")

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    # Load class dictionary (if exists)
    try:
        # Opening the class dictionary file in text mode (with UTF-8 encoding)
        with open(r"D:\sportspersonclassifier\server\artifacts\class_dictionary.json", "r", encoding="utf-8") as f:
            __class_name_to_number = json.load(f)
            __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}
    except FileNotFoundError:
        print("Class dictionary file not found.")
    except Exception as e:
        print(f"Error loading class dictionary: {e}")

    global __model
    if __model is None:
        try:
            # Opening the model file in binary mode ('rb') since it's a binary file
            with open(r"D:\sportspersonclassifier\server\artifacts\saved_model.pkl", 'rb') as f:
                __model = joblib.load(f)
        except FileNotFoundError:
            print("Model file not found.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    print("loading saved artifacts...done")

def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    try:
        encoded_data = b64str.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier(r"D:\sportspersonclassifier\open cv\haar cascade\haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(r"D:\sportspersonclassifier\open cv\haar cascade\haarcascade_eye.xml")

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    if img is None:
        print("Error: Image not loaded correctly.")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    
    return cropped_faces

def get_b64_test_image_for_virat():
    b64_file_path = "b64.txt"
    if os.path.exists(b64_file_path):
        try:
            # Open the file as a binary file
            with open(b64_file_path, "rb") as f:
                # Read the raw bytes and convert them to a base64 string
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"Error reading base64 file: {e}")
            return None
    else:
        print(f"Error: {b64_file_path} not found.")
        return None

if __name__ == '__main__':
    load_saved_artifacts()

    image_path = r"D:\sportspersonclassifier\server\test_image\serena1.jpg"
    
    if os.path.exists(image_path):
        print(f"Image exists at {image_path}. Proceeding with classification.")
        print(classify_image(None, image_path))
    else:
        print(f"Error: Image not found at {image_path}")
