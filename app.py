# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import base64
# import io
# import os
# import tempfile
# import numpy as np

# app = Flask(__name__)
# CORS(app)

# # ─── Image Emotion Detection ───────────────────────────────────────────────────

# def analyze_image_emotion(image_source, source_type="file"):
#     """
#     Analyze emotion from image using DeepFace.
#     source_type: 'file' (path), 'array' (numpy), 'base64' (str)
#     """
#     try:
#         from deepface import DeepFace
#         import cv2

#         if source_type == "base64":
#             img_data = base64.b64decode(image_source)
#             nparr = np.frombuffer(img_data, np.uint8)
#             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#             if img is None:
#                 return None, "Could not decode image data"
#             # Save to temp file for DeepFace
#             with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
#                 cv2.imwrite(tmp.name, img)
#                 image_source = tmp.name
#                 source_type = "file"

#         result = DeepFace.analyze(
#             img_path=image_source,
#             actions=["emotion"],
#             enforce_detection=True,
#             silent=True
#         )

#         if isinstance(result, list):
#             result = result[0]

#         dominant = result["dominant_emotion"]
#         emotions = result["emotion"]
#         confidence = round(emotions[dominant], 1)

#         # Clean up temp file if created
#         if source_type == "file" and "tmp" in str(image_source):
#             try:
#                 os.unlink(image_source)
#             except:
#                 pass

#         return {
#             "dominant_emotion": dominant,
#             "confidence": confidence,
#             "all_emotions": {k: round(v, 1) for k, v in emotions.items()}
#         }, None

#     except Exception as e:
#         err = str(e)
#         if "Face could not be detected" in err or "No face" in err:
#             return None, "No face detected in the image. Please use a clear photo with a visible face."
#         if "ModuleNotFoundError" in err or "No module named" in err:
#             return None, f"Missing dependency: {err}"
#         return None, f"Image analysis error: {err}"


# # ─── Audio Emotion Detection ───────────────────────────────────────────────────

# AUDIO_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]

# def extract_audio_features(file_path):
#     """Extract MFCC + pitch + energy features from audio file."""
#     import librosa
#     y, sr = librosa.load(file_path, sr=22050, mono=True)

#     if len(y) < sr * 0.5:
#         raise ValueError("Audio too short (< 0.5 seconds)")

#     # MFCCs
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#     mfcc_mean = np.mean(mfccs, axis=1)
#     mfcc_std  = np.std(mfccs,  axis=1)

#     # Chroma
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#     chroma_mean = np.mean(chroma, axis=1)

#     # Spectral contrast
#     contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#     contrast_mean = np.mean(contrast, axis=1)

#     # RMS energy
#     rms = librosa.feature.rms(y=y)
#     energy = float(np.mean(rms))

#     # Zero crossing rate
#     zcr = librosa.feature.zero_crossing_rate(y)
#     zcr_mean = float(np.mean(zcr))

#     # Pitch (fundamental frequency)
#     pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
#     pitch_vals = pitches[magnitudes > np.median(magnitudes)]
#     avg_pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0

#     features = np.concatenate([
#         mfcc_mean, mfcc_std, chroma_mean, contrast_mean,
#         [energy, zcr_mean, avg_pitch]
#     ])
#     return features, y, sr


# def predict_audio_emotion(features, y, sr):
#     """
#     Heuristic emotion classification from audio features.
#     Replace this with a trained model (.pkl / .h5) for production.
#     """
#     energy   = features[-3]
#     zcr      = features[-2]
#     pitch    = features[-1]
#     mfcc_var = float(np.var(features[:40]))

#     scores = {
#         "neutral":   0.2,
#         "happy":     0.0,
#         "sad":       0.0,
#         "angry":     0.0,
#         "fearful":   0.0,
#         "disgusted": 0.0,
#         "surprised": 0.0,
#     }

#     # High energy + high ZCR + high pitch → happy or surprised
#     if energy > 0.05 and pitch > 200:
#         scores["happy"]     += 0.35
#         scores["surprised"] += 0.20

#     # High energy + low pitch + high ZCR → angry
#     if energy > 0.06 and zcr > 0.1 and pitch < 180:
#         scores["angry"] += 0.40

#     # Low energy + low pitch + low ZCR → sad
#     if energy < 0.02 and pitch < 150:
#         scores["sad"] += 0.40

#     # Low energy + high pitch variation → fearful
#     if energy < 0.03 and mfcc_var > 500:
#         scores["fearful"] += 0.30

#     # High ZCR alone → disgusted or angry
#     if zcr > 0.15:
#         scores["disgusted"] += 0.20
#         scores["angry"]     += 0.10

#     # Moderate everything → neutral
#     if 0.02 <= energy <= 0.05 and 0.05 <= zcr <= 0.12:
#         scores["neutral"] += 0.30

#     # Normalize
#     total = sum(scores.values()) or 1
#     scores = {k: round((v / total) * 100, 1) for k, v in scores.items()}
#     dominant = max(scores, key=scores.__getitem__)
#     return dominant, scores


# # ─── Routes ───────────────────────────────────────────────────────────────────

# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "ok", "message": "Emotion Detection API running"})


# @app.route("/api/detect/image", methods=["POST"])
# def detect_image():
#     try:
#         data = request.get_json(silent=True)

#         # ── URL input ──
#         if data and data.get("url"):
#             import urllib.request
#             url = data["url"]
#             with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
#                 urllib.request.urlretrieve(url, tmp.name)
#                 result, error = analyze_image_emotion(tmp.name, "file")
#             try:
#                 os.unlink(tmp.name)
#             except:
#                 pass
#             if error:
#                 return jsonify({"success": False, "error": error}), 400
#             return jsonify({"success": True, "result": result})

#         # ── Base64 (webcam) ──
#         if data and data.get("image"):
#             b64 = data["image"]
#             if "," in b64:
#                 b64 = b64.split(",", 1)[1]
#             result, error = analyze_image_emotion(b64, "base64")
#             if error:
#                 return jsonify({"success": False, "error": error}), 400
#             return jsonify({"success": True, "result": result})

#         # ── File upload ──
#         if "file" in request.files:
#             file = request.files["file"]
#             if file.filename == "":
#                 return jsonify({"success": False, "error": "No file selected"}), 400
#             allowed = {"jpg", "jpeg", "png", "bmp", "webp"}
#             ext = file.filename.rsplit(".", 1)[-1].lower()
#             if ext not in allowed:
#                 return jsonify({"success": False, "error": f"Unsupported format. Use: {', '.join(allowed)}"}), 400
#             with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
#                 file.save(tmp.name)
#                 result, error = analyze_image_emotion(tmp.name, "file")
#             try:
#                 os.unlink(tmp.name)
#             except:
#                 pass
#             if error:
#                 return jsonify({"success": False, "error": error}), 400
#             return jsonify({"success": True, "result": result})

#         return jsonify({"success": False, "error": "No image data provided"}), 400

#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)}), 500


# @app.route("/api/detect/audio", methods=["POST"])
# def detect_audio():
#     try:
#         if "file" not in request.files:
#             return jsonify({"success": False, "error": "No audio file provided"}), 400

#         file = request.files["file"]
#         if file.filename == "":
#             return jsonify({"success": False, "error": "No file selected"}), 400

#         allowed = {"wav", "mp3", "ogg", "webm", "m4a", "flac"}
#         ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "wav"
#         if ext not in allowed:
#             return jsonify({"success": False, "error": f"Unsupported audio format. Use: {', '.join(allowed)}"}), 400

#         with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
#             file.save(tmp.name)
#             tmp_path = tmp.name

#         try:
#             features, y, sr = extract_audio_features(tmp_path)
#             dominant, scores = predict_audio_emotion(features, y, sr)
#         except ValueError as ve:
#             os.unlink(tmp_path)
#             return jsonify({"success": False, "error": str(ve)}), 400
#         except Exception as e:
#             os.unlink(tmp_path)
#             return jsonify({"success": False, "error": f"Audio processing error: {str(e)}"}), 500
#         finally:
#             try:
#                 os.unlink(tmp_path)
#             except:
#                 pass

#         return jsonify({
#             "success": True,
#             "result": {
#                 "dominant_emotion": dominant,
#                 "confidence": scores[dominant],
#                 "all_emotions": scores
#             }
#         })

#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import os
import tempfile
import numpy as np

app = Flask(__name__)
CORS(app)

# ─── Image Emotion Detection ───────────────────────────────────────────────────

def analyze_image_emotion(image_source, source_type="file"):
    """
    Analyze emotion from image using DeepFace.
    source_type: 'file' (path), 'array' (numpy), 'base64' (str)
    """
    try:
        from deepface import DeepFace
        import cv2

        if source_type == "base64":
            img_data = base64.b64decode(image_source)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None, "Could not decode image data"
            # Save to temp file for DeepFace
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, img)
                image_source = tmp.name
                source_type = "file"

        result = DeepFace.analyze(
            img_path=image_source,
            actions=["emotion"],
            enforce_detection=True,
            silent=True
        )

        if isinstance(result, list):
            result = result[0]

        dominant = result["dominant_emotion"]
        emotions = result["emotion"]
        confidence = round(emotions[dominant], 1)

        # Clean up temp file if created
        if source_type == "file" and "tmp" in str(image_source):
            try:
                os.unlink(image_source)
            except:
                pass

        return {
            "dominant_emotion": dominant,
            "confidence": confidence,
            "all_emotions": {k: round(v, 1) for k, v in emotions.items()}
        }, None

    except Exception as e:
        err = str(e)
        if "Face could not be detected" in err or "No face" in err:
            return None, "No face detected in the image. Please use a clear photo with a visible face."
        if "ModuleNotFoundError" in err or "No module named" in err:
            return None, f"Missing dependency: {err}"
        return None, f"Image analysis error: {err}"


# ─── Audio Emotion Detection ───────────────────────────────────────────────────

AUDIO_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]

def extract_audio_features(file_path):
    """Extract MFCC + pitch + energy features from audio file."""
    import librosa
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    if len(y) < sr * 0.5:
        raise ValueError("Audio too short (< 0.5 seconds)")

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std  = np.std(mfccs,  axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    # RMS energy
    rms = librosa.feature.rms(y=y)
    energy = float(np.mean(rms))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = float(np.mean(zcr))

    # Pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.median(magnitudes)]
    avg_pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0

    features = np.concatenate([
        mfcc_mean, mfcc_std, chroma_mean, contrast_mean,
        [energy, zcr_mean, avg_pitch]
    ])
    return features, y, sr


def predict_audio_emotion(features, y, sr):
    """
    Heuristic emotion classification from audio features.
    Replace this with a trained model (.pkl / .h5) for production.
    """
    energy   = features[-3]
    zcr      = features[-2]
    pitch    = features[-1]
    mfcc_var = float(np.var(features[:40]))

    scores = {
        "neutral":   0.2,
        "happy":     0.0,
        "sad":       0.0,
        "angry":     0.0,
        "fearful":   0.0,
        "disgusted": 0.0,
        "surprised": 0.0,
    }

    # High energy + high ZCR + high pitch → happy or surprised
    if energy > 0.05 and pitch > 200:
        scores["happy"]     += 0.35
        scores["surprised"] += 0.20

    # High energy + low pitch + high ZCR → angry
    if energy > 0.06 and zcr > 0.1 and pitch < 180:
        scores["angry"] += 0.40

    # Low energy + low pitch + low ZCR → sad
    if energy < 0.02 and pitch < 150:
        scores["sad"] += 0.40

    # Low energy + high pitch variation → fearful
    if energy < 0.03 and mfcc_var > 500:
        scores["fearful"] += 0.30

    # High ZCR alone → disgusted or angry
    if zcr > 0.15:
        scores["disgusted"] += 0.20
        scores["angry"]     += 0.10

    # Moderate everything → neutral
    if 0.02 <= energy <= 0.05 and 0.05 <= zcr <= 0.12:
        scores["neutral"] += 0.30

    # Normalize
    total = sum(scores.values()) or 1
    scores = {k: round((v / total) * 100, 1) for k, v in scores.items()}
    dominant = max(scores, key=scores.__getitem__)
    return dominant, scores


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Emotion Detection API running"})


@app.route("/api/detect/image", methods=["POST"])
def detect_image():
    try:
        data = request.get_json(silent=True)

        # ── URL input ──
        if data and data.get("url"):
            import urllib.request
            url = data["url"]
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                urllib.request.urlretrieve(url, tmp.name)
                result, error = analyze_image_emotion(tmp.name, "file")
            try:
                os.unlink(tmp.name)
            except:
                pass
            if error:
                return jsonify({"success": False, "error": error}), 400
            return jsonify({"success": True, "result": result})

        # ── Base64 (webcam) ──
        if data and data.get("image"):
            b64 = data["image"]
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            result, error = analyze_image_emotion(b64, "base64")
            if error:
                return jsonify({"success": False, "error": error}), 400
            return jsonify({"success": True, "result": result})

        # ── File upload ──
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"success": False, "error": "No file selected"}), 400
            allowed = {"jpg", "jpeg", "png", "bmp", "webp"}
            ext = file.filename.rsplit(".", 1)[-1].lower()
            if ext not in allowed:
                return jsonify({"success": False, "error": f"Unsupported format. Use: {', '.join(allowed)}"}), 400
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                file.save(tmp.name)
                result, error = analyze_image_emotion(tmp.name, "file")
            try:
                os.unlink(tmp.name)
            except:
                pass
            if error:
                return jsonify({"success": False, "error": error}), 400
            return jsonify({"success": True, "result": result})

        return jsonify({"success": False, "error": "No image data provided"}), 400

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/detect/audio", methods=["POST"])
def detect_audio():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        allowed = {"wav", "mp3", "ogg", "webm", "m4a", "flac"}
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "wav"
        if ext not in allowed:
            return jsonify({"success": False, "error": f"Unsupported audio format. Use: {', '.join(allowed)}"}), 400

        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            features, y, sr = extract_audio_features(tmp_path)
            dominant, scores = predict_audio_emotion(features, y, sr)
        except ValueError as ve:
            os.unlink(tmp_path)
            return jsonify({"success": False, "error": str(ve)}), 400
        except Exception as e:
            os.unlink(tmp_path)
            return jsonify({"success": False, "error": f"Audio processing error: {str(e)}"}), 500
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

        return jsonify({
            "success": True,
            "result": {
                "dominant_emotion": dominant,
                "confidence": scores[dominant],
                "all_emotions": scores
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
