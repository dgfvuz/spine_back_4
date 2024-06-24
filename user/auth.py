from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, AuthenticationFailed
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import exceptions
import time

from django.utils.translation import gettext_lazy as _

from .models import CustomUser
from django.contrib.auth import authenticate


class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    """
    自定义登录认证，使用自有用户表
    """

    username_field = "account"

    def validate(self, attrs):
        authenticate_kwargs = {
            self.username_field: attrs[self.username_field],
            "password": attrs["password"],
        }
        print(authenticate_kwargs)
        user = authenticate(**authenticate_kwargs)

        if user is None or not user.is_active:

            raise exceptions.AuthenticationFailed(
                _("No active account found with the given credentials")
            )

        refresh = self.get_token(user)
        timestamp = refresh.access_token.payload["exp"]
        time_local = time.localtime(int(timestamp))
        expire_time = time.strftime("%Y-%m-%d %H:%M:%S", time_local)

        data = {
            "account": user.account,
            "name": user.name,
            "email": user.email,
            "description": user.description,
            "is_active": user.is_active,
            "is_staff": user.is_staff,
            "is_superuser": user.is_superuser,
            # avatar字段需要加上服务器地址
            "token": str(refresh.access_token),
            "refresh": str(refresh),
            "expire": expire_time,
        }
        return data


class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer
