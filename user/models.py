from django.contrib.auth.models import (
    AbstractBaseUser,
    BaseUserManager,
    PermissionsMixin,
)
from django.db import models
from .config import avatar_folder, default_avatar


class CustomUserManager(BaseUserManager):
    def create_user(self, account, password=None, **extra_fields):
        if not account:
            raise ValueError("The account field must be set")
        user = self.model(account=account, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, account, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        return self.create_user(account, password, **extra_fields)


class CustomUser(AbstractBaseUser, PermissionsMixin):
    account = models.CharField(max_length=30, primary_key=True, unique=True)
    avatar = models.ImageField(upload_to=avatar_folder, default=default_avatar)
    name = models.CharField(max_length=30, blank=True)
    email = models.EmailField(max_length=100, blank=True)
    description = models.CharField(max_length=100, blank=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=True)

    objects = CustomUserManager()

    USERNAME_FIELD = "account"
    REQUIRED_FIELDS = []

    def __str__(self):
        return self.account
