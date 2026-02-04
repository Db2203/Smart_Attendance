import os
import pickle
import numpy as np
import pandas as pd
import cv2
import face_recognition as fr
from PIL import Image, ImageEnhance
from config import IMAGE_ENHANCEMENT, FACE_RECOGNITION, PATHS


def preprocess_image(image):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting normalization
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Apply PIL enhancements
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(IMAGE_ENHANCEMENT['brightness'])

    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(IMAGE_ENHANCEMENT['contrast'])

    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(IMAGE_ENHANCEMENT['sharpness'])

    return np.array(pil_img)


def load_student_encodings(student_csv=None, force_refresh=False):
    if student_csv is None:
        student_csv = PATHS['student_csv']
    encodings_pkl = PATHS['encodings_pkl']

    need_refresh = force_refresh

    if not os.path.exists(encodings_pkl):
        need_refresh = True
    else:
        csv_mtime = os.path.getmtime(student_csv)
        pickle_mtime = os.path.getmtime(encodings_pkl)
        if csv_mtime > pickle_mtime:
            need_refresh = True

    df = pd.read_csv(student_csv, dtype=str)
    df["Reg No"] = df["Reg No"].str.strip()
    reg_no_to_name = dict(zip(df["Reg No"], df["Name"]))

    if need_refresh:
        encodings_dict = precompute_student_encodings(df)
    else:
        with open(encodings_pkl, "rb") as f:
            encodings_dict = pickle.load(f)
        encodings_dict = {reg.strip(): enc for reg, enc in encodings_dict.items()}

    return encodings_dict, reg_no_to_name


def precompute_student_encodings(df):
    encodings_dict = {}

    for _, row in df.iterrows():
        stud_paths = row['File Paths'].split(',')
        all_encodings = []

        for image_path in stud_paths:
            image_path = image_path.strip()
            if not os.path.exists(image_path):
                print(f"[Warning] Image file not found: {image_path}")
                continue
            try:
                image = fr.load_image_file(image_path)
                image = preprocess_image(image)
                encodings = fr.face_encodings(image, model=FACE_RECOGNITION['model'])
                if encodings:
                    all_encodings.extend(encodings)
            except Exception as e:
                print(f"[Warning] Error processing {image_path}: {e}")

        encodings_dict[row['Reg No']] = all_encodings

    with open(PATHS['encodings_pkl'], "wb") as f:
        pickle.dump(encodings_dict, f)

    print("[Log] Student encodings precomputed and saved.")
    return encodings_dict


def recognize_faces_in_image(image_path, student_encodings, reg_no_to_name):
    logs = []

    if not os.path.exists(image_path):
        logs.append(f"[Error] Image file not found: {image_path}")
        return set(), logs, [], None

    try:
        unknown_image = fr.load_image_file(image_path)
        unknown_image = preprocess_image(unknown_image)
        face_locations = fr.face_locations(unknown_image)
        unknown_encodings = fr.face_encodings(unknown_image,
                                              known_face_locations=face_locations,
                                              model=FACE_RECOGNITION['model'])
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
            if not known_encodings:
                continue
            # Convert and validate encodings
            valid_encodings = []
            for enc in known_encodings:
                if not isinstance(enc, np.ndarray):
                    enc = np.array(enc)
                if enc.ndim != 1:
                    enc = enc.flatten()
                valid_encodings.append(enc)
            if not valid_encodings:
                continue
            # Average encodings for more robust matching
            avg_encoding = np.mean(valid_encodings, axis=0)
            distance = fr.face_distance([avg_encoding], unknown_encoding)[0]
            if distance < best_distance:
                best_distance = distance
                best_match = reg_no

        if best_distance < FACE_RECOGNITION['threshold']:
            recognized_reg_nos.add(best_match)
        elif best_distance < confirmation_threshold:
            candidate_face_location = face_locations[i] if i < len(face_locations) else None
            close_match_candidates.append((unknown_encoding, best_match, best_distance, candidate_face_location))
        else:
            logs.append(f"Unknown face with distance: {best_distance:.3f}")

    return recognized_reg_nos, logs, close_match_candidates, unknown_image
