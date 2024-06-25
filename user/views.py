from django.shortcuts import render

# Create your views here.
from rest_framework import generics
from rest_framework.permissions import AllowAny, IsAuthenticated
from .superuser import IsSuperUser
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .auth import MyTokenObtainPairView
from .models import CustomUser
from .serializers import CustomUserSerializer
from .updateauth import canUpdate
from django.http import FileResponse
import user.config as config
from django.contrib.auth import authenticate
from rest_framework import status
from rest_framework.response import Response
# 创建用户
class CustomUserCreate(generics.CreateAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = CustomUserSerializer
    permission_classes = [IsAuthenticated, IsSuperUser]


# 得到用户列表
class CustomUserList(generics.ListAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = CustomUserSerializer
    permission_classes = [IsAuthenticated, IsSuperUser]
    # 设置分页,每页12个
    # pagination_class = CustomPagination
    # def get_queryset(self):
    #     queryset = CustomUser.objects.all()
    #     account = self.request.query_params.get('account', None)
    #     name = self.request.query_params.get('name', None)
    #     is_active = self.request.query_params.get('is_active', None)
    #     is_staff = self.request.query_params.get('is_staff', None)
    #     is_superuser = self.request.query_params.get('is_superuser', None)


    #     if account is not None:
    #         queryset = queryset.filter(account=account)
    #     return queryset



# 得到用户详情
class CustomUserDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = CustomUserSerializer
    permission_classes = [IsAuthenticated, canUpdate]
    lookup_field = 'account'
    # 对password字段设置一个description


    # 设置is_superuser更改权限
    def perform_update(self, serializer):
        if self.request.user.is_superuser:
            serializer.save()
        elif self.request.user.account == self.kwargs['account']:
            serializer.save(is_superuser=False)
    
    # 这里添加一个update方法，用于更新密码,必须提供old_password字段
    def update(self, request, *args, **kwargs):
        # 如果是超级用户，直接更新
        if request.user.is_superuser:
            return super().update(request, *args, **kwargs)
        # 如果是用户自己，需要提供旧密码
        user = self.get_object()
        # 检查是否提供了old_password和new_password
        old_password = request.data.get('old_password')
        password = request.data.get('password')
        if password:
            if not old_password:
                return Response({'error': 'Please provide old password'}, status=status.HTTP_400_BAD_REQUEST)
            # 使用authenticate方法验证old_password
            # 从请求的路径中获取用户名 'users/<str:account>/'
            account = kwargs['account']
            authenticate_kwargs = {'account' : account, 'password': old_password}
            if authenticate(**authenticate_kwargs):
                # 如果验证通过，按照正常逻辑更新密码
                return super().update(request, *args, **kwargs)
            else:
                # 如果old_password验证失败，返回错误信息
                return Response({'error': 'Incorrect old password'}, status=status.HTTP_400_BAD_REQUEST)
        else:
            # 如果没有提供old_password或new_password，按照正常逻辑处理其他更新
            return super().update(request, *args, **kwargs)


class CustomTokenObtainPairView(MyTokenObtainPairView):
    permission_classes = [AllowAny]

class CustomTokenRefreshView(TokenRefreshView):
    permission_classes = [AllowAny]


def show_image(request,file_name):
    print(file_name)
    image_file = config.avatar_folder + file_name
    print(image_file)
    return FileResponse(open(image_file, 'rb'))