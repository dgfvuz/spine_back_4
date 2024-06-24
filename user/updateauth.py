# 鉴权，判断是否是超级用户或者是用户自己
from rest_framework import permissions

class canUpdate(permissions.BasePermission):
    """
    Custom permission to only allow superusers to access the view.
    """

    def has_object_permission(self, request, view, obj):
        return request.user and (request.user.is_superuser or request.user == obj)