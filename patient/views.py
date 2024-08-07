from django.shortcuts import render
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from .models import Patient
from .serializers import PatientSerializer, PatientRegionSerializer
from .pagination import CustomPagination
from django.utils.dateparse import parse_datetime

# Create your views here.
class CreatePatient(generics.CreateAPIView):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer
    permission_classes = [IsAuthenticated]


class ListPatient(generics.ListAPIView):
    serializer_class = PatientSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPagination
    def get_queryset(self):
        queryset = Patient.objects.all().order_by('-update_time')
        # 获取查询参数
        name = self.request.query_params.get('name', None)
        gender = self.request.query_params.get('gender', None)
        age__gt = self.request.query_params.get('age__gt', None)
        age = self.request.query_params.get('age', None)
        age__lt = self.request.query_params.get('age__lt', None)
        region = self.request.query_params.get('region', None)
        address = self.request.query_params.get('address', None)
        start_time = self.request.query_params.get('start_time', None)
        end_time = self.request.query_params.get('end_time', None)

        # 解析开始时间和结束时间
        if start_time is not None:
            start_time = parse_datetime(start_time)
            if start_time:
                queryset = queryset.filter(update_time__gte=start_time)

        if end_time is not None:
            end_time = parse_datetime(end_time)
            if end_time:
                queryset = queryset.filter(update_time__lte=end_time)
        # 根据姓名过滤
        if name is not None:
            queryset = queryset.filter(name__icontains=name)
        # 根据性别过滤
        if gender is not None:
            queryset = queryset.filter(gender=gender)
        # 根据年龄过滤
        if age__gt is not None:
            queryset = queryset.filter(age__gt=age__gt)
        if age is not None:
            queryset = queryset.filter(age=age)
        # 根据年龄过滤
        if age__lt is not None:
            queryset = queryset.filter(age__lt=age__lt)
        # 根据地区过滤
        if region is not None:
            queryset = queryset.filter(region__icontains=region)
        # 根据地址过滤
        if address is not None:
            queryset = queryset.filter(address__icontains=address)
        return queryset


class DetailPatient(generics.RetrieveUpdateDestroyAPIView):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'id'

class ListPatientRegion(generics.ListAPIView):
    serializer_class = PatientRegionSerializer
    permission_classes = [IsAuthenticated]
    def get_queryset(self):
        queryset = Patient.objects.values('region').distinct()
        return queryset