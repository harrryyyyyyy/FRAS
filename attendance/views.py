import base64
import cv2
import numpy as np
import threading
from django.shortcuts import render
from django.http import JsonResponse
from django.utils import timezone
from .models import User_Detail, Attendance, Organization
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from .face_model import face_app

# Global cache for embeddings
known_faces_cache = {}
cache_lock = threading.Lock()

# --------- Load Known Faces ---------
def load_known_faces():
    """Pre-load all known face embeddings into cache"""
    global known_faces_cache
    with cache_lock:
        if known_faces_cache:  # Cache already populated
            return

        print("Loading known faces embeddings...")
        
        users_with_embeddings = User_Detail.objects.filter(embedding__isnull=False)

        count = 0
        for user in users_with_embeddings:
            try:
                embedding_array = np.array(user.embedding, dtype=np.float32)
                norm = np.linalg.norm(embedding_array)
                if norm == 0:
                    print(f"Skipping user {user.userId}, invalid embedding.")
                    continue
                normalized_embedding = embedding_array / norm
                known_faces_cache[user.userId] = {
                    'vendor': user.organization,
                    'embedding': normalized_embedding, # Store the normalized embedding
                    'name': str(user) # Get the display name from the __str__ method
                }
                count += 1
            except Exception as e:
                print(f"Error loading embedding for user {user.userId}: {e}")
        
        print(f"Loaded {count} known faces from database.")

load_known_faces()

def get_face_embedding(img):
    try:
        faces = face_app.get(img)
        if not faces:
            print("Mark Attendance: No face detected in image.")
            return None
        
        embedding = faces[0].embedding
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        return embedding / norm
    except Exception as e:
        print(f"Error during new embedding generation: {e}")
        return None

def find_best_match(new_embedding, cache, threshold):
    best_score = -1
    best_user_id = None
    
    with cache_lock:
        for user_id, data in cache.items():
            known_emb = data['embedding'] 
            score = np.dot(new_embedding, known_emb)
            
            if score > best_score and score > threshold:
                best_score = score
                best_user_id = user_id
                
    return best_user_id, best_score

# mark
# @csrf_exempt
# def mark_attendance(request):
    # if request.method == 'POST':
    #     image_data = request.POST.get('image')
    #     if not image_data:
    #         return JsonResponse({'message': 'No image data provided.'}, status=400)

    #     try:
    #         format, imgstr = image_data.split(';base64,')
    #         image_bytes = base64.b64decode(imgstr)
    #         np_arr = np.frombuffer(image_bytes, np.uint8)
    #         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
    #         if img is None:
    #             raise ValueError("Could not decode image.")
    #     except Exception as e:
    #         print(f"Image decode error: {e}")
    #         return JsonResponse({'success': False, 'message': 'Invalid image format.'}, status=400)

    #     new_embedding = get_face_embedding(img)
    #     if new_embedding is None:
    #         return JsonResponse({'success': False, 'message': 'No face detected or poor image quality.'}, status=400)

    #     matched_user_id, score = find_best_match(new_embedding, known_faces_cache, threshold=0.6)
        
    #     if matched_user_id:
    #         try:
    #             user = User_Detail.objects.get(userId=matched_user_id)
                
    #             # --- Check-in/Check-out Logic ---
    #             today = timezone.now().date()
    #             existing_attendance = Attendance.objects.filter(user=user, timestamp__date=today).order_by('-timestamp').first()
                
    #             status = not existing_attendance.isCheckin if existing_attendance else True
    #             output_status = 'Check-In' if status else 'Check-Out'
    #             output_message = 'Welcome' if status else 'Thank You'

    #             Attendance.objects.create(user=user, isCheckin=status)
                
    #             user_name = known_faces_cache[matched_user_id]['name']
    #             return JsonResponse({'success': True, 'message': f'{output_status}, {user_name}! Status: {output_status}, Score:{score}'})

    #         except User_Detail.DoesNotExist:
    #              return JsonResponse({'success': False, 'message': 'Match found but user not in DB.'}, status=404)
    #     else:
    #         return JsonResponse({'success': False, 'message': 'Face not recognized.'}, status=404)

    # return render(request, 'attendance/mark_attendance.html')



@csrf_exempt
def mark_attendance(request):
    if request.method == 'POST':
        image_data = request.POST.get('image')
        if not image_data:
            return JsonResponse({'success': False, 'message': 'No image data provided.'}, status=400)

        try:
            format, imgstr = image_data.split(';base64,')
            image_bytes = base64.b64decode(imgstr)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image.")
        except Exception as e:
            print(f"Image decode error: {e}")
            return JsonResponse({'success': False, 'message': 'Invalid image format.'}, status=400)

        # --- Detect all faces ---
        try:
            faces = face_app.get(img)
            if not faces:
                return JsonResponse({'success': False, 'message': 'No faces detected in image.'}, status=400)
        except Exception as e:
            print(f"Face detection error: {e}")
            return JsonResponse({'success': False, 'message': 'Error detecting faces.'}, status=400)

        recognized_users = []

        # --- Process each detected face ---
        for face in faces:
            embedding = face.embedding
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            embedding = embedding / norm

            matched_user_id, score = find_best_match(embedding, known_faces_cache, threshold=0.6)
            if not matched_user_id:
                continue

            try:
                user = User_Detail.objects.get(userId=matched_user_id)
                today = timezone.now().date()
                existing_attendance = Attendance.objects.filter(user=user, timestamp__date=today).order_by('-timestamp').first()
                
                status = not existing_attendance.isCheckin if existing_attendance else True
                Attendance.objects.create(user=user, isCheckin=status)
                
                recognized_users.append({
                    'user': known_faces_cache[matched_user_id]['name'],
                    'status': 'Check-In' if status else 'Check-Out'
                })
            except User_Detail.DoesNotExist:
                print(f"User with ID {matched_user_id} not found in DB.")
                continue

        if recognized_users:
            names = ', '.join([f"{u['user']} ({u['status']})" for u in recognized_users])
            return JsonResponse({'success': True, 'message': f'Attendance marked for: {names}'})
        else:
            return JsonResponse({'success': False, 'message': 'No known faces recognized.'}, status=404)

    return render(request, 'attendance/mark_attendance.html')

# Register
@csrf_exempt
def register_user(request):
    if request.method == 'POST':
        firstname = request.POST.get('firstname') or request.POST.get('name')
        lastname = request.POST.get('lastname', '')
        phone = request.POST.get('phone', '')
        email = request.POST.get('email', '')
        organization = request.POST.get('organization', '')
        image_data = request.POST.get('image')

        if not firstname or not image_data:
            return JsonResponse({'message': 'First name or image missing!'}, status=400)
        
        if User_Detail.objects.filter(phone=phone).exists():
            return JsonResponse({'message': 'Duplicate mobile number!'}, status=400)
        
        if organization.strip().upper() == 'STATE BANK OF INDIA (SBI)':
            isVendor = False
        else:
            isVendor = True

        format, imgstr = image_data.split(';base64,') 
        ext = format.split('/')[-1]  # e.g., png
        img_file = ContentFile(base64.b64decode(imgstr), name=f'{firstname}.{ext}')

        # Generate a unique userId (optional)
        import uuid
        userId = str(uuid.uuid4())

        # Create and save the user
        user = User_Detail(
            userId=userId,
            firstname=firstname,
            lastname=lastname,
            phone=phone,
            email=email,
            organization=organization,
            isVendor=isVendor,
            profile_pic=img_file
        )
        user.save()

        if user.embedding:
            with cache_lock:
                embedding_array = np.array(user.embedding, dtype=np.float32)

                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    normalized_embedding = embedding_array / norm
                    known_faces_cache[user.userId] = {
                        'vendor': user.organization,
                        'embedding': normalized_embedding,
                        'name': str(user)
                    }
                    print(f"User {user.userId} added to live cache.")

        return JsonResponse({'message': f'User {firstname} registered successfully!'})

    organizations = Organization.objects.all()
    return render(request, 'attendance/register.html', {'organizations': organizations})

# add vendor
@csrf_exempt 
def add_organization(request):
    if request.method == "POST":
        org_name = request.POST.get("name", "").strip()
        if not org_name:
            return JsonResponse({"success": False, "message": "Organization name is required."})

        org, created = Organization.objects.get_or_create(name=org_name)
        if created:
            message = "Organization added successfully!"
        else:
            message = "Organization already exists."

        return JsonResponse({"success": True, "message": message, "name": org.name})
    
    return JsonResponse({"success": False, "message": "Invalid request method."})