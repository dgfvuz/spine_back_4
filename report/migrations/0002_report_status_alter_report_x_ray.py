# Generated by Django 5.0.6 on 2024-06-29 05:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("report", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="report",
            name="status",
            field=models.CharField(default="生成中", max_length=20),
        ),
        migrations.AlterField(
            model_name="report",
            name="X_ray",
            field=models.ImageField(blank=True, upload_to="X_ray/"),
        ),
    ]