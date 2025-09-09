from django.db import models

class Organization(models.Model):
    name = models.CharField(max_length=200)
    website = models.URLField(blank=True)
    source = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)
    
    def __str__(self):
        return self.name
