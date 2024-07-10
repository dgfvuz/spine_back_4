from django.shortcuts import render
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from .models import Collect
from .serializers import CollectSerializer
from patient.pagination import CustomPagination
from rest_framework.exceptions import ValidationError
# Create your views here.

class CollectCreateView(generics.CreateAPIView):
    queryset = Collect.objects.all()
    serializer_class = CollectSerializer
    permission_classes = [IsAuthenticated]
    def perform_create(self, serializer):
        patient = serializer.validated_data.get('patient')
        user = serializer.validated_data.get('user')
        # 检查数据库中是否已经存在该患者的信息
        collect = Collect.objects.filter(patient=patient, user=user)
        if collect.exists():
            raise ValidationError('用户已经收集了该患者的信息')
        super().perform_create(serializer)

class CollectListView(generics.ListAPIView):
    serializer_class = CollectSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPagination
    def get_queryset(self):
        queryset = Collect.objects.all().order_by('-timestamp')
        # 获取查询参数
        user = self.request.query_params.get('user', None)
        if user is not None:
            queryset = queryset.filter(user=user)
        return queryset
    
class CollectDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Collect.objects.all()
    serializer_class = CollectSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'id'