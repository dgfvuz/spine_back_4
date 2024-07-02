"""
URL configuration for app project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from rest_framework.documentation import include_docs_urls
from user.views import ShowImageView
from user.config import avatar_folder
from report.views import ShowXRayImageView
from report.config import X_ray_folder

urlpatterns = [
    path("admin/", admin.site.urls),
    path("user/", include("user.urls")),
    path("patient/", include("patient.urls")),
    path("report/", include("report.urls")),
    path("advice/", include("advice.urls")),
    path("collect/", include("collect.urls")),
    path("docs/", include_docs_urls(title="API")),
    path(avatar_folder +'<str:file_name>', ShowImageView.as_view(), name='avatar'),
    path(X_ray_folder +'<str:file_name>', ShowXRayImageView.as_view(), name='xray'),
]

