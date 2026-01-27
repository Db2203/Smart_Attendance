"""Face Recognition Service for Web Application.

Wraps the existing face_recognition_module.py functionality for use in Flask.
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image, ImageEnhance
import face_recognition as fr
from flask import current_app

# Add src directory to path to import existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


class FaceRecognitionService:
    """Service class for face recognition operations."""

    def __init__(self):
        self.encodings = {}
        self.threshold = 0.52          # Slightly relaxed to catch more matches
        self.confirmation_margin = 0.12  # Wider margin for close matches
        self.model = 'cnn'

    def load_encodings_from_file(self, pkl_path=None):
        """Load face encodings from pickle file."""
        if pkl_path is None:
            pkl_path = current_app.config.get('ENCODINGS_FILE')

        if pkl_path and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self.encodings = pickle.load(f)
            return True
        return False

    def load_encodings_from_db(self, students):
        """Load face encodings from database Student objects."""
        self.encodings = {}
        for student in students:
            if student.face_encoding:
                try:
                    encoding = pickle.loads(student.face_encoding)
                    if student.reg_no not in self.encodings:
                        self.encodings[student.reg_no] = []
                    if isinstance(encoding, list):
                        self.encodings[student.reg_no].extend(encoding)
                    else:
                        self.encodings[student.reg_no].append(encoding)
                except Exception as e:
                    print(f"Error loading encoding for {student.reg_no}: {e}")

    def preprocess_image(self, image):
        """Apply image enhancements for better recognition."""
        pil_img = Image.fromarray(image)

        # Apply brightness enhancement
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.2)

        # Apply contrast enhancement
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.2)

        # Apply sharpness enhancement
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.1)

        return np.array(pil_img)

    def process_image(self, image_path):
        """Process an image and return recognized faces.

        Args:
            image_path: Path to the image file

        Returns:
            dict with keys:
                - recognized: list of (reg_no, name, confidence) tuples
                - close_matches: list of (reg_no, name, confidence, face_location) tuples
                - unknown_count: number of unrecognized faces
                - error: error message if any
        """
        result = {
            'recognized': [],
            'close_matches': [],
            'unknown_count': 0,
            'error': None
        }

        if not os.path.exists(image_path):
            result['error'] = f"Image file not found: {image_path}"
            return result

        try:
            # Load and preprocess image
            image = fr.load_image_file(image_path)
            image = self.preprocess_image(image)

            # Detect faces
            face_locations = fr.face_locations(image)
            if not face_locations:
                result['error'] = "No faces detected in the image"
                return result

            # Get encodings for detected faces
            unknown_encodings = fr.face_encodings(
                image,
                known_face_locations=face_locations,
                model=self.model
            )

            if not unknown_encodings:
                result['error'] = "Could not encode detected faces"
                return result

            # Match each face against known encodings
            confirmation_threshold = self.threshold + self.confirmation_margin

            for i, unknown_encoding in enumerate(unknown_encodings):
                best_match = None
                best_distance = 1.0

                for reg_no, known_encodings in self.encodings.items():
                    for known_encoding in known_encodings:
                        if not isinstance(known_encoding, np.ndarray):
                            known_encoding = np.array(known_encoding)
                        if known_encoding.ndim != 1:
                            known_encoding = known_encoding.flatten()

                        distance = fr.face_distance([known_encoding], unknown_encoding)[0]
                        if distance < best_distance:
                            best_distance = distance
                            best_match = reg_no

                if best_distance < self.threshold:
                    # Confident match
                    confidence = round((1 - best_distance) * 100, 1)
                    result['recognized'].append({
                        'reg_no': best_match,
                        'confidence': confidence,
                        'distance': round(best_distance, 3)
                    })
                elif best_distance < confirmation_threshold:
                    # Close match - needs confirmation
                    confidence = round((1 - best_distance) * 100, 1)
                    face_loc = face_locations[i] if i < len(face_locations) else None
                    result['close_matches'].append({
                        'reg_no': best_match,
                        'confidence': confidence,
                        'distance': round(best_distance, 3),
                        'face_location': face_loc
                    })
                else:
                    # Unknown face
                    result['unknown_count'] += 1

        except Exception as e:
            result['error'] = f"Error processing image: {str(e)}"

        return result

    def get_face_count(self, image_path):
        """Get the number of faces in an image."""
        try:
            image = fr.load_image_file(image_path)
            face_locations = fr.face_locations(image)
            return len(face_locations)
        except Exception:
            return 0

    def generate_encoding_from_file(self, image_path):
        """Generate face encoding from a single image file.

        Args:
            image_path: Path to the image file

        Returns:
            tuple: (encoding_bytes, error_message)
                - encoding_bytes: Pickled encoding if successful, None if failed
                - error_message: Error description if failed, None if successful
        """
        if not os.path.exists(image_path):
            return None, f"Image file not found: {image_path}"

        try:
            # Load and preprocess image
            image = fr.load_image_file(image_path)
            image = self.preprocess_image(image)

            # Detect faces
            face_locations = fr.face_locations(image)
            if not face_locations:
                return None, "No face detected in the image"

            if len(face_locations) > 1:
                return None, f"Multiple faces ({len(face_locations)}) detected. Please use a photo with only one face."

            # Generate encoding
            encodings = fr.face_encodings(
                image,
                known_face_locations=face_locations,
                model=self.model
            )

            if not encodings:
                return None, "Could not generate face encoding"

            return encodings[0], None

        except Exception as e:
            return None, f"Error processing image: {str(e)}"

    def generate_encodings_from_files(self, image_paths):
        """Generate face encodings from multiple image files.

        Args:
            image_paths: List of paths to image files

        Returns:
            tuple: (encodings_list, errors_list)
                - encodings_list: List of numpy arrays (successful encodings)
                - errors_list: List of error messages for failed images
        """
        encodings = []
        errors = []

        for path in image_paths:
            encoding, error = self.generate_encoding_from_file(path)
            if encoding is not None:
                encodings.append(encoding)
            if error:
                errors.append(f"{os.path.basename(path)}: {error}")

        return encodings, errors

    def merge_encodings(self, existing_encoding_bytes, new_encodings):
        """Merge new encodings with existing ones.

        Args:
            existing_encoding_bytes: Existing pickled encodings (or None)
            new_encodings: List of new numpy array encodings

        Returns:
            bytes: Pickled list of all encodings
        """
        all_encodings = []

        # Load existing encodings if any
        if existing_encoding_bytes:
            try:
                existing = pickle.loads(existing_encoding_bytes)
                if isinstance(existing, list):
                    all_encodings.extend(existing)
                else:
                    all_encodings.append(existing)
            except Exception:
                pass  # Ignore corrupt existing data

        # Add new encodings
        all_encodings.extend(new_encodings)

        return pickle.dumps(all_encodings)
