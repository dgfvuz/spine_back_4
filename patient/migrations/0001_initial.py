# Generated by Django 5.0.6 on 2024-06-25 06:12

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Patient",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                ("name", models.CharField(max_length=50)),
                ("age", models.IntegerField()),
                ("region", models.CharField(max_length=100)),
                ("address", models.CharField(max_length=100)),
                ("phone", models.CharField(max_length=20)),
                ("id_card", models.CharField(max_length=20)),
                ("gender", models.CharField(max_length=10)),
                ("medical_history", models.TextField()),
                ("allergy_history", models.TextField()),
                ("create_time", models.DateTimeField(auto_now_add=True)),
                ("update_time", models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
