import os
import pickle
import concurrent.futures
import numpy as np
import pandas as pd
import face_recognition as fr
from PIL import Image, ImageEnhance
from my_config import IMAGE_ENHANCEMENT, FACE_RECOGNITION


def preprocess_image(image):
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(IMAGE_ENHANCEMENT['brightness'])
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(IMAGE_ENHANCEMENT['contrast'])
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(IMAGE_ENHANCEMENT['sharpness'])
    return np.array(pil_img)


def load_student_encodings(student_csv='Student.csv'):
    df = pd.read_csv(student_csv)
    reg_no_to_name = dict(zip(df['Reg No'], df['Name']))
    if os.path.exists("student_encodings.pkl"):
        with open("student_encodings.pkl", "rb") as f:
            encodings_dict = pickle.load(f)
    else:
        encodings_dict = precompute_student_encodings(df)
    return encodings_dict, reg_no_to_name


def precompute_student_encodings(df):
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
        encodings_dict[row['Reg No']] = [enc for sublist in results for enc in sublist]

    with open("student_encodings.pkl", "wb") as f:
        pickle.dump(encodings_dict, f)

    print("[Log] Student encodings precomputed and saved.")
    return encodings_dict


def recognize_faces_in_image(image_path, student_encodings, reg_no_to_name):
    logs = []
    try:
        unknown_image = fr.load_image_file(image_path)
        unknown_image = preprocess_image(unknown_image)
        face_locations = fr.face_locations(unknown_image)
        unknown_encodings = fr.face_encodings(unknown_image, known_face_locations=face_locations, model=FACE_RECOGNITION['model'])
    except Exception as e:
        logs.append(f"[Error] Failed to process image: {e}")
        return set(), logs, [], None

    if not unknown_encodings:
        logs.append("[Error] No faces detected in the image.")
        return set(), logs, [], unknown_image

    recognized_reg_nos = set()
    close_match_candidates = []

    confirmation_margin = FACE_RECOGNITION.get('confirmation_margin', 0.1)
    confirmation_threshold = FACE_RECOGNITION['threshold'] + confirmation_margin

    for i, unknown_encoding in enumerate(unknown_encodings):
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
        elif best_distance < confirmation_threshold:
            candidate_face_location = face_locations[i] if i < len(face_locations) else None
            close_match_candidates.append((unknown_encoding, best_match, best_distance, candidate_face_location))
        else:
            logs.append(f"Unknown face with distance: {best_distance:.3f}")

    return recognized_reg_nos, logs, close_match_candidates, unknown_image
