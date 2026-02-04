"""Face Recognition Service for Web Application.

Uses InsightFace models (SCRFD for detection, ArcFace for recognition) with
ONNX Runtime GPU acceleration for fast face recognition.
"""

import os
import sys
import pickle
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from numpy.linalg import norm

# Add CUDA DLL paths before importing onnxruntime
# This enables GPU acceleration if nvidia-cuda-runtime-cu12 and nvidia-cudnn-cu12 are installed
def _setup_cuda_paths():
    """Add CUDA DLL paths to environment for ONNX Runtime GPU support."""
    try:
        import site
        site_packages = site.getsitepackages()[0]
        if not os.path.exists(site_packages):
            for sp in site.getsitepackages():
                if os.path.exists(sp) and 'site-packages' in sp:
                    site_packages = sp
                    break

        nvidia_paths = [
            os.path.join(site_packages, 'nvidia', 'cublas', 'bin'),
            os.path.join(site_packages, 'nvidia', 'cuda_runtime', 'bin'),
            os.path.join(site_packages, 'nvidia', 'cudnn', 'bin'),
        ]

        existing_paths = [p for p in nvidia_paths if os.path.exists(p)]
        if existing_paths:
            current_path = os.environ.get('PATH', '')
            new_paths = ';'.join(existing_paths)
            os.environ['PATH'] = new_paths + ';' + current_path
    except Exception:
        pass  # Silently ignore if CUDA packages not installed

_setup_cuda_paths()

import onnxruntime

# Add src directory to path to import existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


def softmax(z):
    """Compute softmax values for each set of scores in z."""
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


def distance2bbox(points, distance):
    """Decode distance prediction to bounding box."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    """Decode distance prediction to keypoints."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFDDetector:
    """SCRFD face detector using ONNX Runtime with GPU."""

    def __init__(self, model_path, providers=None):
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = onnxruntime.InferenceSession(model_path, sess_options, providers=providers)
        self.taskname = 'detection'
        self.center_cache = {}
        self.nms_threshold = 0.4
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        self.input_name = input_cfg.name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.use_kps = False
        self._num_anchors = 1
        num_outputs = len(self.output_names)

        if num_outputs == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif num_outputs == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif num_outputs == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif num_outputs == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc] * stride
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def detect(self, img, threshold=0.5, input_size=(640, 640), max_num=0):
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]

        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, threshold)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale

        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self._nms(pre_det)
        det = pre_det[keep, :]

        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        return det, kpss

    def _nms(self, dets):
        thresh = self.nms_threshold
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


class ArcFaceRecognizer:
    """ArcFace face recognizer using ONNX Runtime with GPU."""

    def __init__(self, model_path, providers=None):
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = onnxruntime.InferenceSession(model_path, sess_options, providers=providers)
        self.taskname = 'recognition'

        # Determine normalization parameters
        import onnx
        model = onnx.load(model_path)
        graph = model.graph
        find_sub = False
        find_mul = False
        for node in graph.node[:8]:
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True

        if find_sub and find_mul:
            self.input_mean = 0.0
            self.input_std = 1.0
        else:
            self.input_mean = 127.5
            self.input_std = 127.5

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_size = tuple(input_cfg.shape[2:4][::-1])
        self.output_names = [o.name for o in self.session.get_outputs()]

    def get_embedding(self, img):
        """Get face embedding from aligned face image."""
        assert img.shape[2] == 3
        input_size = tuple(img.shape[0:2][::-1])
        assert input_size == self.input_size, f"Expected {self.input_size}, got {input_size}"

        blob = cv2.dnn.blobFromImage(
            img, 1.0 / self.input_std, input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        return net_outs[0].flatten()


def align_face(img, kps, image_size=112):
    """Align face using 5 keypoints."""
    # Standard face template for 112x112
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    if image_size != 112:
        src = src * image_size / 112

    dst = kps.astype(np.float32)

    # Estimate affine transform
    tform = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)[0]
    aligned = cv2.warpAffine(img, tform, (image_size, image_size), borderValue=0.0)
    return aligned


class FaceRecognitionService:
    """Service class for face recognition operations using InsightFace with GPU."""

    def __init__(self):
        self.encodings = {}
        self.threshold = 0.35  # Cosine similarity threshold (different from Euclidean)
        self.confirmation_margin = 0.08
        self.det_size = (640, 640)
        self.det_threshold = 0.5

        # Models will be loaded on first use
        self._detector = None
        self._recognizer = None
        self._models_loaded = False

        # Model paths
        self.models_dir = os.path.expanduser('~/.insightface/models/buffalo_l')

    def _load_models(self):
        """Load ONNX models with GPU support."""
        if self._models_loaded:
            return

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # Load detector (SCRFD)
        det_model_path = os.path.join(self.models_dir, 'det_10g.onnx')
        if os.path.exists(det_model_path):
            self._detector = SCRFDDetector(det_model_path, providers)
            print(f"[InsightFace] Loaded detector: {det_model_path}")
        else:
            raise FileNotFoundError(f"Detection model not found: {det_model_path}")

        # Load recognizer (ArcFace)
        rec_model_path = os.path.join(self.models_dir, 'w600k_r50.onnx')
        if os.path.exists(rec_model_path):
            self._recognizer = ArcFaceRecognizer(rec_model_path, providers)
            print(f"[InsightFace] Loaded recognizer: {rec_model_path}")
        else:
            raise FileNotFoundError(f"Recognition model not found: {rec_model_path}")

        # Check actual provider used by the session
        actual_providers = self._detector.session.get_providers()
        if 'CUDAExecutionProvider' in actual_providers:
            print("[InsightFace] GPU acceleration active (CUDA)")
        else:
            print("[InsightFace] Running on CPU (GPU unavailable or CUDA 12.x + cuDNN 9.x not installed)")

        self._models_loaded = True

    def load_encodings_from_file(self, pkl_path=None):
        """Load face encodings from pickle file."""
        from flask import current_app
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
        # Apply CLAHE for lighting normalization
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Apply PIL enhancements
        pil_img = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.1)

        return np.array(pil_img)

    def _cosine_similarity(self, feat1, feat2):
        """Compute cosine similarity between two feature vectors."""
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        return np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))

    def process_image(self, image_path):
        """Process an image and return recognized faces.

        Args:
            image_path: Path to the image file

        Returns:
            dict with keys:
                - recognized: list of dicts with reg_no, confidence, distance
                - close_matches: list of dicts with reg_no, confidence, distance, face_location
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
            # Load models if not loaded
            self._load_models()

            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                result['error'] = f"Could not read image: {image_path}"
                return result

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.preprocess_image(image)

            # Detect faces
            bboxes, kpss = self._detector.detect(image, threshold=self.det_threshold,
                                                   input_size=self.det_size)

            if bboxes.shape[0] == 0:
                result['error'] = "No faces detected in the image"
                return result

            # Get embeddings for detected faces
            confirmation_threshold = self.threshold - self.confirmation_margin

            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, 0:4]
                kps = kpss[i] if kpss is not None else None

                if kps is None:
                    # Extract face without alignment if no keypoints
                    x1, y1, x2, y2 = map(int, bbox)
                    face_img = image[y1:y2, x1:x2]
                    face_img = cv2.resize(face_img, (112, 112))
                else:
                    # Align face using keypoints
                    face_img = align_face(image, kps)

                # Get embedding
                embedding = self._recognizer.get_embedding(face_img)
                embedding = embedding / norm(embedding)  # L2 normalize

                # Match against known encodings
                best_match = None
                best_similarity = -1.0

                for reg_no, known_encodings in self.encodings.items():
                    if not known_encodings:
                        continue

                    # Convert and validate encodings
                    valid_encodings = []
                    for enc in known_encodings:
                        if not isinstance(enc, np.ndarray):
                            enc = np.array(enc)
                        if enc.ndim != 1:
                            enc = enc.flatten()
                        # Normalize
                        enc = enc / norm(enc)
                        valid_encodings.append(enc)

                    if not valid_encodings:
                        continue

                    # Use median for outlier-robust averaging
                    avg_encoding = np.median(valid_encodings, axis=0)
                    avg_encoding = avg_encoding / norm(avg_encoding)

                    similarity = self._cosine_similarity(avg_encoding, embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = reg_no

                face_location = (int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[0]))  # top, right, bottom, left

                if best_similarity > self.threshold:
                    # Confident match - convert to Python float for JSON serialization
                    confidence = float(round(best_similarity * 100, 1))
                    result['recognized'].append({
                        'reg_no': best_match,
                        'confidence': confidence,
                        'distance': float(round(1 - best_similarity, 3))
                    })
                elif best_similarity > confirmation_threshold:
                    # Close match - needs confirmation
                    confidence = float(round(best_similarity * 100, 1))
                    result['close_matches'].append({
                        'reg_no': best_match,
                        'confidence': confidence,
                        'distance': float(round(1 - best_similarity, 3)),
                        'face_location': face_location
                    })
                else:
                    # Unknown face
                    result['unknown_count'] += 1

        except Exception as e:
            import traceback
            result['error'] = f"Error processing image: {str(e)}\n{traceback.format_exc()}"

        return result

    def get_face_count(self, image_path):
        """Get the number of faces in an image."""
        try:
            self._load_models()
            image = cv2.imread(image_path)
            if image is None:
                return 0
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, _ = self._detector.detect(image, threshold=self.det_threshold,
                                               input_size=self.det_size)
            return bboxes.shape[0]
        except Exception:
            return 0

    def generate_encoding_from_file(self, image_path):
        """Generate face encoding from a single image file.

        Args:
            image_path: Path to the image file

        Returns:
            tuple: (encoding, error_message)
                - encoding: numpy array if successful, None if failed
                - error_message: Error description if failed, None if successful
        """
        if not os.path.exists(image_path):
            return None, f"Image file not found: {image_path}"

        try:
            self._load_models()

            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return None, f"Could not read image: {image_path}"

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.preprocess_image(image)

            # Detect faces
            bboxes, kpss = self._detector.detect(image, threshold=self.det_threshold,
                                                   input_size=self.det_size)

            if bboxes.shape[0] == 0:
                return None, "No face detected in the image"

            if bboxes.shape[0] > 1:
                return None, f"Multiple faces ({bboxes.shape[0]}) detected. Please use a photo with only one face."

            # Get keypoints
            kps = kpss[0] if kpss is not None else None

            if kps is None:
                bbox = bboxes[0, 0:4]
                x1, y1, x2, y2 = map(int, bbox)
                face_img = image[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (112, 112))
            else:
                face_img = align_face(image, kps)

            # Get embedding
            embedding = self._recognizer.get_embedding(face_img)
            embedding = embedding / norm(embedding)  # L2 normalize

            return embedding, None

        except Exception as e:
            import traceback
            return None, f"Error processing image: {str(e)}\n{traceback.format_exc()}"

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
