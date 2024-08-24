from django.db import models

class ImageModel(models.Model):
    image = models.ImageField(upload_to='images/', help_text="Upload and image file")

    def __str__(self):
        return self.image.name