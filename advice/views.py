from django.shortcuts import render
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from .models import Advice
from .serializers import AdviceSerializer


class AdviceCreateView(generics.CreateAPIView):
    queryset = Advice.objects.all()
    serializer_class = AdviceSerializer
    permission_classes = [IsAuthenticated]


class AdviceListView(generics.ListAPIView):
    serializer_class = AdviceSerializer
    permission_classes = [IsAuthenticated]
    def get_queryset(self):
        queryset = Advice.objects.all().order_by('-timestamp')
        return queryset