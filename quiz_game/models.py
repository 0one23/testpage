from django.db import models

# Create your models here.

class Question(models.Model):
    question_id = models.AutoField(primary_key=True)
    question = models.TextField()
    level = models.IntegerField(default=1)
    category = models.TextField()
    answer = models.IntegerField()
    choice1 = models.TextField()
    choice2 = models.TextField()
    choice3 = models.TextField()
    choice4 = models.TextField()
    explanation = models.TextField()
    reference = models.TextField()

class Player(models.Model):
    player_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=10)
    password = models.CharField(max_length=20, default="testuser")
    BMI = models.IntegerField()
    athretic = models.IntegerField()
    sleep = models.IntegerField()
    knowledge = models.IntegerField()