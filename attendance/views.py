import base64
import cv2
import numpy as np
import threading
import uuid
from django.shortcuts import render
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from .models import User_Detail, Attendance, Organization
from .face_model import face_app

# GLOBALS & LOCKS
known_faces_cache = {}
cache_lock = threading.Lock()

def json_response(success, message, code=200, **extra):
    payload = {"success": success, "message": message}
    payload.update(extra)
    return JsonResponse(payload, status=code)

def decode_image(image_data):
    try:
        _, img_str = image_data.split(';base64,')
        img_bytes = base64.b64decode(img_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img if img is not None else None
    except Exception:
        return None

def load_known_faces():
    """Load all user embeddings into memory once."""
    global known_faces_cache
    with cache_lock:
        if known_faces_cache:
            return  # Already loaded

        print("Loading face embeddings from database...")
        users = User_Detail.objects.filter(embedding__isnull=False)
        for user in users:
            try:
                emb = np.array(user.embedding, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm == 0:
                    continue
                known_faces_cache[user.userId] = {
                    "embedding": emb / norm,
                    "name": str(user),
                    "org": user.organization,
                }
            except Exception as e:
                print(f"Error loading {user}: {e}")

        print(f"Loaded {len(known_faces_cache)} embeddings into cache.")

load_known_faces()

def get_face_embeddings(img):
    try:
        faces = face_app.get(img)
        return [
            f.embedding / np.linalg.norm(f.embedding)
            for f in faces
            if np.linalg.norm(f.embedding) > 0
        ]
    except Exception as e:
        print(f"Embedding error: {e}")
        return []


def find_best_match(embedding, threshold=0.6):
    with cache_lock:
        if not known_faces_cache:
            return None, None

        user_ids = list(known_faces_cache.keys())
        all_embs = np.stack([known_faces_cache[u]["embedding"] for u in user_ids])
        sims = np.dot(all_embs, embedding)
        idx = np.argmax(sims)
        if sims[idx] >= threshold:
            return user_ids[idx], sims[idx]
        return None, None

def mark_user_attendance(user):
    today = timezone.now().date()
    last = Attendance.objects.filter(user=user, timestamp__date=today).order_by('-timestamp').first()
    is_checkin = not last.isCheckin if last else True
    Attendance.objects.create(user=user, isCheckin=is_checkin)
    return "Check-In" if is_checkin else "Check-Out"

# Recogniton logic
def process_image_for_attendance(img, mode="single", wave_detected=True):
    embeddings = get_face_embeddings(img)
    if not embeddings:
        return json_response(False, "No faces detected.", 404)

    if mode == "wave" and not wave_detected:
        return json_response(False, "No wave detected.")

    recognized = []
    for emb in ([embeddings[0]] if mode == "single" else embeddings):
        user_id, score = find_best_match(emb)
        if not user_id:
            continue

        try:
            user = User_Detail.objects.get(userId=user_id)
            status = mark_user_attendance(user)
            output_message = 'Welcome!' if status == 'Check-In' else 'Thank You!'
            recognized.append(f"{output_message} {user} ({status})")
        except User_Detail.DoesNotExist:
            continue

    if recognized:
        return json_response(True, f"{', '.join(recognized)}")
    return json_response(False, "No known faces recognized.", 404)


# ENDPOINTS

# button
@csrf_exempt
def mark_attendance(request):
    if request.method == "GET":
        return render(request, "attendance/mark_attendance.html")

    image_data = request.POST.get("image")
    img = decode_image(image_data)
    if img is None:
        return json_response(False, "Invalid image data.")
    return process_image_for_attendance(img, mode="single")

# touchless
@csrf_exempt
def touchless_mark_attendance(request):
    if request.method == "GET":
        return render(request, "attendance/touchless_mark_attendance.html")

    if request.method == "POST":
        img = decode_image(request.POST.get("image"))
        if img is None:
            return json_response(False, "Invalid image data.")
        return process_image_for_attendance(img, mode="touchless")

@csrf_exempt
# def wave_mark_attendance(request):
#     """Wave gesture — mark everyone visible."""
#     if request.method != "POST":
#         return json_response(False, "Invalid request method.")
#     img = decode_image(request.POST.get("image"))
#     if img is None:
#         return json_response(False, "Invalid image data.")
#     # TODO: Integrate real gesture detection here
#     wave_detected = True
#     return process_image_for_attendance(img, mode="wave", wave_detected=wave_detected)

@csrf_exempt
def wave_mark_attendance(request):
    if request.method == "GET":
        return render(request, "attendance/wave_mark_attendance.html")

    if request.method == "POST":
        image_data = request.POST.get("image")
        img = decode_image(image_data)
        if img is None:
            return json_response(False, "Invalid image data.")
        wave_detected = True
        return process_image_for_attendance(img, mode="wave", wave_detected=wave_detected)

    return json_response(False, "Unsupported request method.")

# USER REGISTRATION
@csrf_exempt
def register_user(request):
    if request.method == "GET":
        orgs = Organization.objects.all()
        return render(request, "attendance/register.html", {"organizations": orgs})

    firstname = request.POST.get("firstname") or request.POST.get("name")
    image_data = request.POST.get("image")
    if not firstname or not image_data:
        return json_response(False, "Name or image missing!")

    phone = request.POST.get("phone", "")
    if User_Detail.objects.filter(phone=phone).exists():
        return json_response(False, "Duplicate mobile number!")

    # Prepare image file
    fmt, imgstr = image_data.split(";base64,")
    ext = fmt.split("/")[-1]
    img_file = ContentFile(base64.b64decode(imgstr), name=f"{firstname}.{ext}")

    org_name = request.POST.get("organization", "").strip()
    is_vendor = org_name.upper() != "STATE BANK OF INDIA (SBI)"

    user = User_Detail.objects.create(
        userId=str(uuid.uuid4()),
        firstname=firstname,
        lastname=request.POST.get("lastname", ""),
        phone=phone,
        email=request.POST.get("email", ""),
        organization=org_name,
        isVendor=is_vendor,
        profile_pic=img_file,
    )

    # Update cache if embedding exists
    if user.embedding:
        emb = np.array(user.embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            with cache_lock:
                known_faces_cache[user.userId] = {
                    "embedding": emb / norm,
                    "name": str(user),
                    "org": user.organization,
                }

    return json_response(True, f"User {firstname} registered successfully!")


# add organization
@csrf_exempt
def add_organization(request):
    if request.method != "POST":
        return json_response(False, "Invalid request method.")
    name = request.POST.get("name", "").strip()
    if not name:
        return json_response(False, "Organization name is required.")
    org, created = Organization.objects.get_or_create(name=name)
    msg = "Organization added successfully!" if created else "Organization already exists."
    return json_response(True, msg, name=org.name)

@csrf_exempt
def find_twin(request):
    if request.method == "GET":
        return render(request, "attendance/find_twin.html")

    if request.method == "POST":
        image_data = request.POST.get("image")
        user_threshold = float(request.POST.get("threshold", 0.6))

        img = decode_image(image_data)
        if img is None:
            return json_response(False, "Invalid image data.")

        embeddings = get_face_embeddings(img)
        if not embeddings:
            return json_response(False, "No faces detected.", 404)

        emb = embeddings[0]

        with cache_lock:
            if not known_faces_cache:
                return json_response(False, "No known faces in system.", 404)

            user_ids = list(known_faces_cache.keys())
            all_embs = np.stack([known_faces_cache[u]["embedding"] for u in user_ids])
            sims = np.dot(all_embs, emb)

            # Exclude self
            max_sim = np.max(sims)
            self_similarity_threshold = min(0.98, max_sim * 0.999)
            mask = sims < self_similarity_threshold
            if not mask.any():
                return json_response(False, "No close match found.", similarity=float(max_sim))

            sims_filtered = sims * mask

            # Get top 2 matches
            top_indices = sims_filtered.argsort()[::-1][:2]
            results = []
            for idx in top_indices:
                score = float(sims_filtered[idx])
                if score <= 0:
                    continue
                best_user_id = user_ids[idx]
                best_user = User_Detail.objects.get(userId=best_user_id)
                photo_url = best_user.profile_pic.url if best_user.profile_pic else None
                results.append({
                    "name": str(best_user),
                    "similarity": round(score * 100, 2),
                    # "photo_url": photo_url
                })

            if not results:
                # fallback: return closest match even if below threshold
                idx = np.argmax(sims_filtered)
                best_user_id = user_ids[idx]
                best_user = User_Detail.objects.get(userId=best_user_id)
                results.append({
                    "name": str(best_user),
                    "similarity": round(float(sims_filtered[idx]) * 100, 2),
                    # "photo_url": best_user.profile_pic.url if best_user.profile_pic else None
                })

            return json_response(
                True,
                "Found your look-alike(s)!",
                lookalikes=results
            )

    return json_response(False, "Unsupported request method.")
