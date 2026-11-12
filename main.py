
import os
import time
import shutil
import tempfile
import base64
import json
import traceback

from pathlib import Path
from paddleocr import PaddleOCR

import pytesseract
from pytesseract import Output

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="OCR Comparison API",
    description="Independent PaddleOCR and Tesseract comparison service",
    version="1.0.0",
)


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MAX_FILE_SIZE = 15 * 1024 * 1024
MAX_IMAGE_SIDE = 2000

TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

TESSERACT_CONFIG = "--oem 3 --psm 6"

DENOISE_STRENGTH = 10
MAX_SKEW_ANGLE = 15
MIN_DESKEW_ANGLE = 0.3

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(
    input_path: str,
    output_path: str,
    grayscale: bool = True,
    resize: bool = True,
    denoising: bool = True,
    deskewing: bool = True,
    contrast: bool = True,
    adaptive_threshold: bool = True,
):
    image = cv2.imread(input_path)

    if image is None:
        raise ValueError("Unable to read uploaded image.")

    timings = {}
    applied = []

    total_start = time.perf_counter()

    def record(name, operation):
        nonlocal image

        start = time.perf_counter()
        image = operation(image)

        timings[name] = round(
            (time.perf_counter() - start) * 1000, 2
        )

        applied.append(name)

    # 1. GRAYSCALE
    if grayscale:
        record(
            "grayscale",
            lambda img: (
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if len(img.shape) == 3
                else img
            ),
        )

    # 2. RESIZE
    if resize:
        def resize_image(img):
            height, width = img.shape[:2]
            longest = max(height, width)

            if longest <= MAX_IMAGE_SIDE:
                return img

            scale = MAX_IMAGE_SIDE / longest

            return cv2.resize(
                img,
                (
                    int(width * scale),
                    int(height * scale),
                ),
                interpolation=cv2.INTER_AREA,
            )

        record("resize", resize_image)

    # 3. DENOISING
    if denoising:
        def denoise_image(img):
            if len(img.shape) == 2:
                return cv2.fastNlMeansDenoising(
                    img,
                    None,
                    DENOISE_STRENGTH,
                    7,
                    21,
                )

            return cv2.fastNlMeansDenoisingColored(
                img,
                None,
                DENOISE_STRENGTH,
                DENOISE_STRENGTH,
                7,
                21,
            )

        record("denoising", denoise_image)

    # 4. DESKEWING
    if deskewing:
        def deskew_image(img):
            gray = (
                img
                if len(img.shape) == 2
                else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            )

            binary = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )[1]

            edges = cv2.Canny(binary, 50, 150, apertureSize=3)

            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=100,
                minLineLength=max(30, img.shape[1] // 5),
                maxLineGap=20,
            )

            if lines is None:
                return img

            angles = []

            for line in lines:
                values = np.asarray(line).reshape(-1)

                if values.size < 4:
                    continue

                x1, y1, x2, y2 = map(float, values[:4])

                angle = np.degrees(
                    np.arctan2(y2 - y1, x2 - x1)
                )

                if (
                    MIN_DESKEW_ANGLE <= abs(angle)
                    <= MAX_SKEW_ANGLE
                ):
                    angles.append(angle)

            if not angles:
                return img

            median_angle = float(np.median(angles))

            height, width = img.shape[:2]
            center = (width // 2, height // 2)

            matrix = cv2.getRotationMatrix2D(
                center,
                median_angle,
                1.0,
            )

            return cv2.warpAffine(
                img,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

        record("deskewing", deskew_image)

    # 5. CONTRAST ENHANCEMENT (CLAHE)
    if contrast:
        def enhance_contrast(img):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2GRAY,
                )
            else:
                gray = img

            clahe = cv2.createCLAHE(
                clipLimit=CLAHE_CLIP_LIMIT,
                tileGridSize=CLAHE_TILE_GRID_SIZE,
            )

            return clahe.apply(gray)

        record("contrast", enhance_contrast)

    # 6. ADAPTIVE THRESHOLDING
    if adaptive_threshold:
        def threshold_image(img):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2GRAY,
                )
            else:
                gray = img

            height, width = gray.shape

            block_size = min(
                ADAPTIVE_BLOCK_SIZE,
                min(height, width),
            )

            if block_size % 2 == 0:
                block_size -= 1

            if block_size < 3:
                return gray

            return cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                ADAPTIVE_C,
            )

        record("adaptive_threshold", threshold_image)

    if not cv2.imwrite(output_path, image):
        raise ValueError("Unable to save processed image.")

    total_time = round(
        (time.perf_counter() - total_start) * 1000, 2
    )

    height, width = image.shape[:2]

    return {
        "image_path": output_path,
        "enabled": True,
        "applied": applied,
        "timings_ms": timings,
        "total_time_ms": total_time,
        "output_dimensions": {
            "width": width,
            "height": height,
        },
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "success": True,
        "message": "OCR Comparison API is running.",
        "version": "1.0.0",
    }


@app.get("/api")
async def api_status():
    return {
        "success": True,
        "service": "OCR Comparison Backend",
        "engines": ["PaddleOCR", "Tesseract"],
        "preprocessing": [
            "grayscale",
            "resize",
            "denoising",
            "deskewing",
            "contrast",
            "adaptive_threshold",
        ],
    }


# --------------------------------------------------
# PREPROCESSING ENDPOINT
# --------------------------------------------------

@app.post("/api/preprocess")
async def preprocess_endpoint(
    file: UploadFile = File(...),
    preprocessing: bool = Form(False),
    grayscale: bool = Form(False),
    resize: bool = Form(False),
    denoising: bool = Form(False),
    deskewing: bool = Form(False),
    contrast: bool = Form(False),
    adaptive_threshold: bool = Form(False),
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG, and WEBP images are supported.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 15 MB upload limit.",
        )

    temp_dir = tempfile.mkdtemp(prefix="ocr_comparison_")

    try:
        original_path = os.path.join(
            temp_dir,
            "original" + extension,
        )

        processed_path = os.path.join(
            temp_dir,
            "processed.png",
        )

        with open(original_path, "wb") as image_file:
            image_file.write(content)

        if not cv2.haveImageReader(original_path):
            raise HTTPException(
                status_code=400,
                detail="Invalid or unsupported image file.",
            )

        original_image = cv2.imread(original_path)

        if original_image is None:
            raise HTTPException(
                status_code=400,
                detail="Unable to decode uploaded image.",
            )

        original_height, original_width = original_image.shape[:2]

        settings = {
            "grayscale": grayscale,
            "resize": resize,
            "denoising": denoising,
            "deskewing": deskewing,
            "contrast": contrast,
            "adaptive_threshold": adaptive_threshold,
        }

        if preprocessing:
            result = preprocess_image(
                original_path,
                processed_path,
                **settings,
            )

            output_path = processed_path
        else:
            result = {
                "enabled": False,
                "applied": [],
                "timings_ms": {},
                "total_time_ms": 0,
                "output_dimensions": {
                    "width": original_width,
                    "height": original_height,
                },
            }

            output_path = original_path

        with open(output_path, "rb") as output_file:
            output_bytes = output_file.read()

        return {
            "success": True,
            "filename": file.filename,
            "original": {
                "width": original_width,
                "height": original_height,
                "preserved": True,
            },
            "preprocessing": result,
            "settings": {
                "enabled": preprocessing,
                **settings,
            },
            "processed_image_base64": base64.b64encode(
                output_bytes
            ).decode("utf-8"),
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# --------------------------------------------------
# PADDLEOCR ENGINE
# --------------------------------------------------

_paddle_ocr = None



def get_paddle_ocr():
    global _paddle_ocr

    if _paddle_ocr is None:
        print("\n" + "=" * 60, flush=True)
        print("[PaddleOCR] MODEL INITIALIZATION STARTED", flush=True)

        start_time = time.perf_counter()

        try:
            print("[PaddleOCR] Loading English OCR model...", flush=True)

            _paddle_ocr = PaddleOCR(
                lang="en",
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="en_PP-OCRv5_mobile_rec",
                text_det_limit_side_len=960,
                text_det_limit_type="max",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

            elapsed = round(
                (time.perf_counter() - start_time) * 1000, 2
            )

            print(
                f"[PaddleOCR] MODEL INITIALIZED SUCCESSFULLY "
                f"({elapsed} ms)",
                flush=True,
            )

        except Exception as error:
            print("[PaddleOCR] MODEL INITIALIZATION FAILED", flush=True)
            print(f"[PaddleOCR] Error: {error}", flush=True)
            traceback.print_exc()
            raise

    return _paddle_ocr



# --------------------------------------------------
# PADDLEOCR RESULT FORMATTER
# --------------------------------------------------

def format_paddle_results(predictions):
    results = []

    for prediction in predictions:
        raw_data = prediction.json

        if callable(raw_data):
            raw_data = raw_data()

        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        data = raw_data.get("res", raw_data)

        texts = data.get("rec_texts", []) or []
        scores = data.get("rec_scores", []) or []

        for index, line in enumerate(texts):
            line = str(line).strip()

            if not line:
                continue

            score = (
                float(scores[index])
                if index < len(scores)
                else 0.0
            )

            confidence = (
                score * 100
                if score <= 1
                else score
            )

            confidence = round(confidence, 2)

            if confidence >= 90:
                status = "HIGH"
            elif confidence >= 70:
                status = "MODERATE"
            else:
                status = "NEEDS_REVIEW"

            results.append({
                "text": line,
                "confidence": confidence,
                "status": status,
                "flagged": confidence < 70,
            })

    return results


# --------------------------------------------------
# TESSERACT RESULT FORMATTER
# --------------------------------------------------

def format_tesseract_results(data):
    grouped = {}

    total_words = len(data["text"])

    for index in range(total_words):
        word = str(data["text"][index]).strip()

        if not word:
            continue

        try:
            confidence = float(data["conf"][index])
        except (ValueError, TypeError):
            continue

        if confidence < 0:
            continue

        key = (
            data["page_num"][index],
            data["block_num"][index],
            data["par_num"][index],
            data["line_num"][index],
        )

        if key not in grouped:
            grouped[key] = {
                "words": [],
                "confidences": [],
            }

        grouped[key]["words"].append(word)
        grouped[key]["confidences"].append(confidence)

    results = []

    for line_data in grouped.values():
        text = " ".join(line_data["words"])

        confidence = round(
            sum(line_data["confidences"])
            / len(line_data["confidences"]),
            2,
        )

        if confidence >= 90:
            status = "HIGH"
        elif confidence >= 70:
            status = "MODERATE"
        else:
            status = "NEEDS_REVIEW"

        results.append({
            "text": text,
            "confidence": confidence,
            "status": status,
            "flagged": confidence < 70,
        })

    return results


# --------------------------------------------------
# SHARED PREPROCESSING SETTINGS
# --------------------------------------------------

def get_preprocessing_settings(
    grayscale,
    resize,
    denoising,
    deskewing,
    contrast,
    adaptive_threshold,
):
    return {
        "grayscale": grayscale,
        "resize": resize,
        "denoising": denoising,
        "deskewing": deskewing,
        "contrast": contrast,
        "adaptive_threshold": adaptive_threshold,
    }


# --------------------------------------------------
# PADDLEOCR API ENDPOINT WITH DEBUG LOGGING
# --------------------------------------------------

@app.post("/api/ocr/paddle")
async def paddle_ocr_endpoint(
    file: UploadFile = File(...),
    preprocessing: bool = Form(False),
    grayscale: bool = Form(False),
    resize: bool = Form(False),
    denoising: bool = Form(False),
    deskewing: bool = Form(False),
    contrast: bool = Form(False),
    adaptive_threshold: bool = Form(False),
):
    request_start = time.perf_counter()

    print("\n" + "=" * 60, flush=True)
    print("[PaddleOCR] NEW REQUEST RECEIVED", flush=True)

    temp_dir = None

    try:
        # STEP 1: VALIDATE FILE
        print("[PaddleOCR] Step 1: Validating uploaded file...", flush=True)

        extension = Path(file.filename or "").suffix.lower()

        print(f"[PaddleOCR] Filename: {file.filename}", flush=True)
        print(f"[PaddleOCR] Extension: {extension}", flush=True)

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, JPEG, PNG, and WEBP images are supported.",
            )

        content = await file.read()

        print(
            f"[PaddleOCR] Uploaded size: {len(content) / 1024:.2f} KB",
            flush=True,
        )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 15 MB upload limit.",
            )

        # STEP 2: SAVE IMAGE
        print("[PaddleOCR] Step 2: Saving uploaded image...", flush=True)

        temp_dir = tempfile.mkdtemp(prefix="paddle_ocr_")

        original_path = os.path.join(
            temp_dir,
            "original" + extension,
        )

        processed_path = os.path.join(
            temp_dir,
            "processed.png",
        )

        with open(original_path, "wb") as image_file:
            image_file.write(content)

        print("[PaddleOCR] Image saved successfully.", flush=True)

        # STEP 3: READ IMAGE
        print("[PaddleOCR] Step 3: Reading image dimensions...", flush=True)

        original_image = cv2.imread(original_path)

        if original_image is None:
            raise HTTPException(
                status_code=400,
                detail="Unable to decode uploaded image.",
            )

        original_height, original_width = original_image.shape[:2]

        print(
            f"[PaddleOCR] Image dimensions: "
            f"{original_width} x {original_height}",
            flush=True,
        )

        # STEP 4: PREPROCESSING
        print("[PaddleOCR] Step 4: Checking preprocessing settings...", flush=True)

        settings = get_preprocessing_settings(
            grayscale,
            resize,
            denoising,
            deskewing,
            contrast,
            adaptive_threshold,
        )

        print(
            f"[PaddleOCR] Preprocessing enabled: {preprocessing}",
            flush=True,
        )
        print(f"[PaddleOCR] Settings: {settings}", flush=True)

        preprocessing_result = {
            "enabled": False,
            "applied": [],
            "timings_ms": {},
            "total_time_ms": 0,
        }

        ocr_input_path = original_path

        if preprocessing:
            print("[PaddleOCR] Starting image preprocessing...", flush=True)

            preprocessing_start = time.perf_counter()

            preprocessing_result = preprocess_image(
                original_path,
                processed_path,
                **settings,
            )

            ocr_input_path = processed_path

            preprocessing_elapsed = round(
                (time.perf_counter() - preprocessing_start) * 1000,
                2,
            )

            print(
                f"[PaddleOCR] Preprocessing completed "
                f"({preprocessing_elapsed} ms)",
                flush=True,
            )
            print(
                f"[PaddleOCR] Applied: {preprocessing_result['applied']}",
                flush=True,
            )

        else:
            print(
                "[PaddleOCR] Preprocessing skipped. Using original image.",
                flush=True,
            )

        # STEP 5: INITIALIZE MODEL
        # CONVERT WEBP TO PNG FOR PADDLEOCR
        if extension == ".webp":
            print(
                "[PaddleOCR] WebP detected. Converting to PNG...",
                flush=True,
            )

            webp_image = cv2.imread(ocr_input_path)

            if webp_image is None:
                raise HTTPException(
                    status_code=400,
                    detail="Unable to convert WebP image for PaddleOCR.",
                )

            paddle_png_path = os.path.join(
                temp_dir,
                "paddle_input.png",
            )

            if not cv2.imwrite(paddle_png_path, webp_image):
                raise HTTPException(
                    status_code=500,
                    detail="Failed to convert WebP image for PaddleOCR.",
                )

            ocr_input_path = paddle_png_path

            print(
                "[PaddleOCR] WebP successfully converted to PNG.",
                flush=True,
            )
        print("[PaddleOCR] Step 5: Getting OCR engine...", flush=True)

        engine = get_paddle_ocr()

        print("[PaddleOCR] OCR engine ready.", flush=True)

        # STEP 6: RUN OCR
        print("[PaddleOCR] Step 6: Starting OCR prediction...", flush=True)
        print(f"[PaddleOCR] Input image: {ocr_input_path}", flush=True)

        ocr_start = time.perf_counter()

        predictions = engine.predict(
            ocr_input_path,
            return_word_box=True,
        )

        ocr_time = round(
            (time.perf_counter() - ocr_start) * 1000,
            2,
        )

        print(
            f"[PaddleOCR] OCR prediction completed ({ocr_time} ms)",
            flush=True,
        )

        # STEP 7: FORMAT RESULTS
        print("[PaddleOCR] Step 7: Formatting OCR results...", flush=True)

        results = format_paddle_results(predictions)

        print(
            f"[PaddleOCR] Detected lines: {len(results)}",
            flush=True,
        )

        extracted_text = "\n".join(
            item["text"] for item in results
        )

        confidences = [
            item["confidence"] for item in results
        ]

        average_confidence = (
            round(sum(confidences) / len(confidences), 2)
            if confidences
            else 0
        )

        total_lines = len(results)

        flagged_lines = sum(
            1 for item in results if item["flagged"]
        )

        print(
            f"[PaddleOCR] Average confidence: {average_confidence}%",
            flush=True,
        )
        print(
            f"[PaddleOCR] Flagged lines: {flagged_lines}",
            flush=True,
        )

        # STEP 8: COMPLETE RESPONSE
        total_time = round(
            (time.perf_counter() - request_start) * 1000,
            2,
        )

        print("[PaddleOCR] REQUEST COMPLETED SUCCESSFULLY", flush=True)
        print(f"[PaddleOCR] Total time: {total_time} ms", flush=True)
        print("=" * 60 + "\n", flush=True)

        return {
            "success": True,
            "engine": "PaddleOCR",
            "filename": file.filename,

            "original_image": {
                "preserved": True,
                "width": original_width,
                "height": original_height,
            },

            "preprocessing": preprocessing_result,

            "settings": {
                "enabled": preprocessing,
                **settings,
            },

            "text": extracted_text,

            "confidence_summary": {
                "average": average_confidence,
                "total_lines": total_lines,
                "flagged_lines": flagged_lines,
            },

            "results": results,

            "processing_time_ms": {
                "preprocessing": preprocessing_result.get(
                    "total_time_ms", 0
                ),
                "ocr": ocr_time,
                "total": round(
                    preprocessing_result.get(
                        "total_time_ms", 0
                    ) + ocr_time,
                    2,
                ),
            },
        }

    except HTTPException as error:
        print(
            f"[PaddleOCR] REQUEST VALIDATION ERROR: {error.detail}",
            flush=True,
        )
        raise

    except Exception as error:
        print("\n[PaddleOCR] REQUEST FAILED", flush=True)
        print(f"[PaddleOCR] Error type: {type(error).__name__}", flush=True)
        print(f"[PaddleOCR] Error message: {error}", flush=True)
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"PaddleOCR processing failed: {str(error)}",
        )

    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("[PaddleOCR] Temporary files cleaned up.", flush=True)


# --------------------------------------------------
# TESSERACT OCR API ENDPOINT WITH DEBUG LOGGING
# --------------------------------------------------

@app.post("/api/ocr/tesseract")
async def tesseract_ocr_endpoint(
    file: UploadFile = File(...),
    preprocessing: bool = Form(False),
    grayscale: bool = Form(False),
    resize: bool = Form(False),
    denoising: bool = Form(False),
    deskewing: bool = Form(False),
    contrast: bool = Form(False),
    adaptive_threshold: bool = Form(False),
):
    request_start = time.perf_counter()

    print("\n" + "=" * 60, flush=True)
    print("[Tesseract] NEW REQUEST RECEIVED", flush=True)

    temp_dir = None

    try:
        # STEP 1: VALIDATE FILE
        print("[Tesseract] Step 1: Validating uploaded file...", flush=True)

        extension = Path(file.filename or "").suffix.lower()

        print(f"[Tesseract] Filename: {file.filename}", flush=True)
        print(f"[Tesseract] Extension: {extension}", flush=True)

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, JPEG, PNG, and WEBP images are supported.",
            )

        content = await file.read()

        print(
            f"[Tesseract] Uploaded size: {len(content) / 1024:.2f} KB",
            flush=True,
        )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 15 MB upload limit.",
            )

        # STEP 2: CHECK TESSERACT
        print("[Tesseract] Step 2: Checking Tesseract installation...", flush=True)
        print(f"[Tesseract] Executable: {TESSERACT_CMD!r}", flush=True)
        print(f"[Tesseract] Path exists: {os.path.exists(TESSERACT_CMD)}", flush=True)
        print(f"[Tesseract] Is file: {os.path.isfile(TESSERACT_CMD)}", flush=True)
        print(f"[Tesseract] Found in PATH: {shutil.which('tesseract')}", flush=True)

        if not os.path.isfile(TESSERACT_CMD):
            raise HTTPException(
                status_code=500,
                detail="Tesseract executable was not found.",
            )

        print("[Tesseract] Executable found.", flush=True)
        print(f"[Tesseract] OCR config: {TESSERACT_CONFIG}", flush=True)

        # STEP 3: SAVE IMAGE
        print("[Tesseract] Step 3: Saving uploaded image...", flush=True)

        temp_dir = tempfile.mkdtemp(prefix="tesseract_ocr_")

        original_path = os.path.join(
            temp_dir,
            "original" + extension,
        )

        processed_path = os.path.join(
            temp_dir,
            "processed.png",
        )

        with open(original_path, "wb") as image_file:
            image_file.write(content)

        print("[Tesseract] Image saved successfully.", flush=True)

        # STEP 4: READ IMAGE
        print("[Tesseract] Step 4: Reading image dimensions...", flush=True)

        original_image = cv2.imread(original_path)

        if original_image is None:
            raise HTTPException(
                status_code=400,
                detail="Unable to decode uploaded image.",
            )

        original_height, original_width = original_image.shape[:2]

        print(
            f"[Tesseract] Image dimensions: "
            f"{original_width} x {original_height}",
            flush=True,
        )

        # STEP 5: PREPROCESSING
        print("[Tesseract] Step 5: Checking preprocessing settings...", flush=True)

        settings = get_preprocessing_settings(
            grayscale,
            resize,
            denoising,
            deskewing,
            contrast,
            adaptive_threshold,
        )

        print(
            f"[Tesseract] Preprocessing enabled: {preprocessing}",
            flush=True,
        )
        print(f"[Tesseract] Settings: {settings}", flush=True)

        preprocessing_result = {
            "enabled": False,
            "applied": [],
            "timings_ms": {},
            "total_time_ms": 0,
        }

        ocr_input_path = original_path

        if preprocessing:
            print("[Tesseract] Starting image preprocessing...", flush=True)

            preprocessing_start = time.perf_counter()

            preprocessing_result = preprocess_image(
                original_path,
                processed_path,
                **settings,
            )

            ocr_input_path = processed_path

            preprocessing_elapsed = round(
                (time.perf_counter() - preprocessing_start) * 1000,
                2,
            )

            print(
                f"[Tesseract] Preprocessing completed "
                f"({preprocessing_elapsed} ms)",
                flush=True,
            )

            print(
                f"[Tesseract] Applied: {preprocessing_result['applied']}",
                flush=True,
            )

            dimensions = preprocessing_result.get(
                "output_dimensions", {}
            )

            print(
                f"[Tesseract] Processed dimensions: "
                f"{dimensions.get('width')} x {dimensions.get('height')}",
                flush=True,
            )

        else:
            print(
                "[Tesseract] Preprocessing skipped. Using original image.",
                flush=True,
            )

        # STEP 6: RUN TESSERACT OCR
        print("[Tesseract] Step 6: Starting OCR...", flush=True)
        print(f"[Tesseract] Input image: {ocr_input_path}", flush=True)
        print("[Tesseract] Language: eng", flush=True)

        ocr_start = time.perf_counter()

        raw_results = pytesseract.image_to_data(
            ocr_input_path,
            lang="eng",
            config=TESSERACT_CONFIG,
            output_type=Output.DICT,
        )

        ocr_time = round(
            (time.perf_counter() - ocr_start) * 1000,
            2,
        )

        print(
            f"[Tesseract] OCR completed successfully ({ocr_time} ms)",
            flush=True,
        )

        # STEP 7: COUNT DETECTED WORDS
        detected_words = sum(
            1
            for word in raw_results["text"]
            if str(word).strip()
        )

        print(
            f"[Tesseract] Detected words before formatting: "
            f"{detected_words}",
            flush=True,
        )

        # STEP 8: FORMAT RESULTS
        print("[Tesseract] Step 8: Formatting OCR results...", flush=True)

        results = format_tesseract_results(raw_results)

        print(
            f"[Tesseract] Detected lines: {len(results)}",
            flush=True,
        )

        extracted_text = "\n".join(
            item["text"] for item in results
        )

        confidences = [
            item["confidence"] for item in results
        ]

        average_confidence = (
            round(sum(confidences) / len(confidences), 2)
            if confidences
            else 0
        )

        total_lines = len(results)

        flagged_lines = sum(
            1 for item in results if item["flagged"]
        )

        print(
            f"[Tesseract] Average confidence: "
            f"{average_confidence}%",
            flush=True,
        )

        print(
            f"[Tesseract] Flagged lines: {flagged_lines}",
            flush=True,
        )

        # STEP 9: COMPLETE REQUEST
        total_time = round(
            (time.perf_counter() - request_start) * 1000,
            2,
        )

        print("[Tesseract] REQUEST COMPLETED SUCCESSFULLY", flush=True)
        print(
            f"[Tesseract] Preprocessing time: "
            f"{preprocessing_result.get('total_time_ms', 0)} ms",
            flush=True,
        )
        print(f"[Tesseract] OCR time: {ocr_time} ms", flush=True)
        print(f"[Tesseract] Total request time: {total_time} ms", flush=True)
        print("=" * 60 + "\n", flush=True)

        return {
            "success": True,
            "engine": "Tesseract",
            "filename": file.filename,

            "original_image": {
                "preserved": True,
                "width": original_width,
                "height": original_height,
            },

            "preprocessing": preprocessing_result,

            "settings": {
                "enabled": preprocessing,
                **settings,
            },

            "text": extracted_text,

            "confidence_summary": {
                "average": average_confidence,
                "total_lines": total_lines,
                "flagged_lines": flagged_lines,
            },

            "results": results,

            "processing_time_ms": {
                "preprocessing": preprocessing_result.get(
                    "total_time_ms", 0
                ),
                "ocr": ocr_time,
                "total": round(
                    preprocessing_result.get(
                        "total_time_ms", 0
                    ) + ocr_time,
                    2,
                ),
            },
        }

    except HTTPException as error:
        print(
            f"[Tesseract] REQUEST VALIDATION ERROR: {error.detail}",
            flush=True,
        )
        raise

    except Exception as error:
        print("\n[Tesseract] REQUEST FAILED", flush=True)
        print(
            f"[Tesseract] Error type: {type(error).__name__}",
            flush=True,
        )
        print(
            f"[Tesseract] Error message: {error}",
            flush=True,
        )
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Tesseract processing failed: {str(error)}",
        )

    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("[Tesseract] Temporary files cleaned up.", flush=True)
