from django.db import models

class Link(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField(unique=True)

    def __str__(self):
        return self.title

