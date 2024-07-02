from django.shortcuts import render
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from .models import Collect
from .serializers import CollectSerializer

# Create your views here.

class CollectCreateView(generics.CreateAPIView):
    queryset = Collect.objects.all()
    serializer_class = CollectSerializer
    permission_classes = [IsAuthenticated]

class CollectListView(generics.ListAPIView):
    serializer_class = CollectSerializer
    permission_classes = [IsAuthenticated]
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