# Generated by Django 3.2.7 on 2021-10-01 03:07

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Player',
            fields=[
                ('player_id', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=10)),
                ('password', models.CharField(default='testuser', max_length=20)),
                ('BMI', models.IntegerField()),
                ('athretic', models.IntegerField()),
                ('sleep', models.IntegerField()),
                ('knowledge', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Question',
            fields=[
                ('question_id', models.AutoField(primary_key=True, serialize=False)),
                ('question', models.TextField()),
                ('level', models.IntegerField(default=1)),
                ('category', models.TextField()),
                ('answer', models.IntegerField()),
                ('choice1', models.TextField()),
                ('choice2', models.TextField()),
                ('choice3', models.TextField()),
                ('choice4', models.TextField()),
                ('explanation', models.TextField()),
                ('reference', models.TextField()),
            ],
        ),
    ]
