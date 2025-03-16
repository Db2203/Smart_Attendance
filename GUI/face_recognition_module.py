# face_recognition_module.py

import os
import pickle
import concurrent.futures
import numpy as np
import pandas as pd
import face_recognition as fr
from PIL import Image, ImageEnhance
from my_config import IMAGE_ENHANCEMENT, FACE_RECOGNITION

def preprocess_image(image):
    """Enhance the image's brightness, contrast, and sharpness."""
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(IMAGE_ENHANCEMENT['brightness'])
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(IMAGE_ENHANCEMENT['contrast'])
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(IMAGE_ENHANCEMENT['sharpness'])
    return np.array(pil_img)

def load_student_encodings(student_csv='Student.csv'):
    """
    Load student information and face encodings.
    Reads a CSV file with columns 'Reg No', 'Name', and 'File Paths'.
    """
    df = pd.read_csv(student_csv)
    reg_no_to_name = dict(zip(df['Reg No'], df['Name']))
    if os.path.exists("student_encodings.pkl"):
        with open("student_encodings.pkl", "rb") as f:
            encodings_dict = pickle.load(f)
    else:
        encodings_dict = precompute_student_encodings(df)
    return encodings_dict, reg_no_to_name

def precompute_student_encodings(df):
    """
    Compute face encodings for each student using concurrency,
    then save the results to a pickle file.
    """
    encodings_dict = {}

    def encode_face(image_path):
        try:
            image = fr.load_image_file(image_path)
            image = preprocess_image(image)
            encodings = fr.face_encodings(image, model=FACE_RECOGNITION['model'])
            return encodings if encodings else []
        except Exception as e:
            print(f"[Warning] Error processing {image_path}: {e}")
            return []

    for _, row in df.iterrows():
        stud_paths = row['File Paths'].split(',')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(encode_face, stud_paths))
        # Flatten the list of lists into a single list of encodings
        encodings_dict[row['Reg No']] = [enc for sublist in results for enc in sublist]

    with open("student_encodings.pkl", "wb") as f:
        pickle.dump(encodings_dict, f)

    print("[Log] Student encodings precomputed and saved.")
    return encodings_dict

def recognize_faces_in_image(image_path, student_encodings, reg_no_to_name):
    """
    Given an image file path, detect and recognize faces.
    Returns a set of recognized registration numbers and logs.
    """
    logs = []
    try:
        unknown_image = fr.load_image_file(image_path)
        unknown_image = preprocess_image(unknown_image)
        unknown_encodings = fr.face_encodings(unknown_image, model=FACE_RECOGNITION['model'])
    except Exception as e:
        logs.append(f"[Error] Failed to process image: {e}")
        return set(), logs

    if not unknown_encodings:
        logs.append("[Error] No faces detected in the image.")
        return set(), logs

    recognized_reg_nos = set()

    for unknown_encoding in unknown_encodings:
        # Ensure the encoding is a 1D numpy array
        if not isinstance(unknown_encoding, np.ndarray):
            unknown_encoding = np.array(unknown_encoding)
        if unknown_encoding.ndim != 1:
            unknown_encoding = unknown_encoding.flatten()

        best_match = None
        best_distance = 1.0

        for reg_no, known_encodings in student_encodings.items():
            distances = []
            for known_encoding in known_encodings:
                if not isinstance(known_encoding, np.ndarray):
                    known_encoding = np.array(known_encoding)
                if known_encoding.ndim != 1:
                    known_encoding = known_encoding.flatten()
                d = fr.face_distance([known_encoding], unknown_encoding)[0]
                distances.append(d)
            if distances:
                student_min_distance = min(distances)
                if student_min_distance < best_distance:
                    best_distance = student_min_distance
                    best_match = reg_no

        if best_distance < FACE_RECOGNITION['threshold']:
            recognized_reg_nos.add(best_match)
        else:
            logs.append(f"Unknown face with distance: {best_distance:.3f}")

    return recognized_reg_nos, logs
