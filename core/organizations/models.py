from django.db import models

class Source(models.Model):
    name = models.CharField(max_length=200)
    url = models.URLField()
    last_scraped = models.DateTimeField(auto_now=True) 
    def __str__(self):
        return self.name

class Organization(models.Model):
    source = models.ForeignKey(Source, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    website = models.URLField(blank=True)
    def __str__(self):
        return self.name

class Program(models.Model):  # THIS WAS MISSING
    from django.db import models

class Source(models.Model):
    name = models.CharField(max_length=200)
    url = models.URLField()
    def __str__(self):
        return self.name

class Organization(models.Model):
    source = models.ForeignKey(Source, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    website = models.URLField(blank=True)
    def __str__(self):
        return self.name

class Program(models.Model):  # THIS WAS MISSING
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    eligibility = models.TextField(blank=True)
    def __str__(self):
        return f"{self.name} ({self.organization})"

class Benefit(models.Model):
    program = models.ForeignKey(Program, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    requirements = models.TextField(blank=True)
    def __str__(self):
        return self.name
