from rest_framework_simplejwt.exceptions import InvalidToken, AuthenticationFailed
from django.utils.translation import gettext as _
from user.models import CustomUser
from rest_framework_simplejwt.authentication import JWTAuthentication


class MyJWTAuthentication(JWTAuthentication):
    """
    修改JWT认证类，返回自定义User表对象
    """

    def get_user(self, validated_token):
        try:
            account = validated_token["account"]
        except KeyError:
            raise InvalidToken(_("Token contained no recognizable user identification"))

        try:
            user = CustomUser.objects.get(**{"account": account})
        except CustomUser.DoesNotExist:
            raise AuthenticationFailed(_("User not found"), code="user_not_found")
        return user
