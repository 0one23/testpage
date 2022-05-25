from django.contrib import admin
from quiz_game.models import Player, Question

# Register your models here.
# テスト用のスーパーユーザー
# name: superuser
# password: healthapp

admin.site.register(Player)
admin.site.register(Question)
