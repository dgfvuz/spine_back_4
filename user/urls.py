from django.urls import path
from .views import CustomUserCreate, CustomTokenObtainPairView, CustomTokenRefreshView, CustomUserList, CustomUserDetail, show_image
from django.conf.urls.static import static
from django.conf import settings
from .config import avatar_folder
urlpatterns = [
    # 登录认证,这里说是登录认证，但是实际上是注册一个token
    path('login/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', CustomTokenRefreshView.as_view(), name='token_refresh'),
    # 用户更新
    path('users/', CustomUserList.as_view(), name='users'),
    path('register/', CustomUserCreate.as_view(), name='register'),
    path('users/<str:account>/', CustomUserDetail.as_view(), name='user'),
    # 用户头像
]
