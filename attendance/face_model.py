from insightface.app import FaceAnalysis

print("Loading InsightFace model (this should only run once)...")

def load_model():
    face_app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection', 'recognition']
    )
    face_app.prepare(
        ctx_id=0,
        det_size=(480, 480)
    )
    return face_app

face_app = load_model()
