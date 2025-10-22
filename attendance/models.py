from django.db import models
import cv2
import numpy as np
import cv2 
from .face_model import face_app

def user_directory_path(instance, filename):
    fname = instance.firstname or "user"
    lname = instance.lastname or ""
    id = instance.userId or 0
    org = instance.organization or "tempOrg"
    
    name, ext = filename.split(".")
    name = fname + "_" + lname + "_" + id
    filename = name.replace(" ", "_") + '.' + ext
    return f'User_Images/{org}/{filename}'

def get_face_embedding(img):
    try:
        faces = face_app.get(img)
        if not faces:
            print("Warning: No face detected in image.")
            return None
        return faces[0].embedding
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return None

class User_Detail(models.Model):

    userId = models.CharField(max_length=200, null=False, default="tempUser", blank=False, unique=True)
    firstname = models.CharField(max_length=200, null=False, blank=False)
    lastname = models.CharField(max_length=200, null=True, blank=True)
    phone = models.CharField(max_length=200, null=True, blank=False, unique=True)
    email = models.EmailField(max_length=200, null=True, blank=True)
    organization = models.CharField(max_length=200, null=True, blank=True)
    isVendor = models.BooleanField(max_length=200, null=True, blank=True)
    profile_pic = models.ImageField(upload_to=user_directory_path, null=True, blank=True)
    
    embedding = models.JSONField(null=True, blank=True)

    def __str__(self):
        lname = self.lastname or ""
        return str(self.firstname + " " + lname)

    def save(self, *args, **kwargs):
        if self.profile_pic and not self.embedding: 
            try:
                self.profile_pic.file.seek(0) 
                image_data = self.profile_pic.file.read()
                np_arr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError("Could not decode image.")

                face_embed = get_face_embedding(img)
                
                if face_embed is not None:
                    self.embedding = face_embed.tolist()
                else:
                    self.embedding = None
                    print(f"Warning: No face found for {self.firstname}")

            except Exception as e:
                print(f"Error generating embedding for {self.firstname}: {e}")
                self.embedding = None

        super().save(*args, **kwargs) 

class Attendance(models.Model):
    user = models.ForeignKey(User_Detail, on_delete=models.CASCADE, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    isCheckin = models.BooleanField(max_length=400, null=True, blank=True)

    def __str__(self):
        if self.user:
            return f"{self.user.userId} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
        return f"Unlinked Attendance - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
    

class Organization(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name
