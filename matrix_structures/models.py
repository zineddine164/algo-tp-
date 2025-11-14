from django.db import models
from datetime import datetime

class TpCode(models.Model):
    title = models.CharField(max_length=200)
    code = models.TextField(default="", blank=True, null=False)
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
