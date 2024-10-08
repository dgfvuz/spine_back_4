import threading
from django.http import FileResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
from django.shortcuts import render
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated,AllowAny
from .models import Report
from .serializers import ReportSerializer, GenerateReportSerializer, ReportAnalysisSerializer
from patient.pagination import CustomPagination
from .config import X_ray_folder
from .model import getResult
from django.http import HttpResponse
from rest_framework import status
from rest_framework.views import APIView
from django.utils.dateparse import parse_datetime
from memory_profiler import profile

class CreateReport(generics.CreateAPIView):
    queryset = Report.objects.all()
    serializer_class = ReportSerializer
    permission_classes = [IsAuthenticated]
    
    def perform_create(self, serializer):
        instance = serializer.save()
        # 新建一个线程，用于生成报告
        thread = threading.Thread(target=generate_report, args=(instance,))
        thread.start()
        
    
def generate_report(instance):
    # 在这里实现报告生成的逻辑
    # 通过调用getResult函数，传入X光片的路径，即可得到结果
    results = getResult(instance.X_ray, instance.report_type)
    instance.results = results
    instance.status = '未审核'
    instance.result = '异常'
    instance.save()    

# 定义HTTP方法为post
class RegenerateReportView(generics.GenericAPIView):
    serializer_class = GenerateReportSerializer
    permission_classes = [IsAuthenticated]

    # @profile
    def post(self, request, *args, **kwargs):
        report_id = kwargs['id']
        try:
            report = Report.objects.get(id=report_id)
        except Report.DoesNotExist:
            return HttpResponse(status=status.HTTP_404_NOT_FOUND)
        # 重新生成报告
        report.status = '生成中'
        report.save()
        thread = threading.Thread(target=generate_report, args=(report,))
        thread.start()
        return HttpResponse(status=status.HTTP_200_OK)


class ListReport(generics.ListAPIView):
    serializer_class = ReportSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPagination
    def get_queryset(self):
        queryset = Report.objects.select_related('patient').order_by('-update_time')     
        # 获取查询参数
        patient = self.request.query_params.get('patient', None)
        start_time = self.request.query_params.get('start_time', None)
        end_time = self.request.query_params.get('end_time', None)
        status = self.request.query_params.get('status', None)
        patient_name = self.request.query_params.get('patient_name', None)
        report_type = self.request.query_params.get('report_type', None)
        if report_type is not None:
            queryset = queryset.filter(report_type=report_type)
        if patient_name is not None:
            # 
            queryset = queryset.filter(patient__name__contains=patient_name)
        if status is not None:
            queryset = queryset.filter(status=status)
        if patient is not None:
            queryset = queryset.filter(patient=patient)

        # 解析开始时间和结束时间
        if start_time is not None:
            start_time = parse_datetime(start_time)
            if start_time:
                queryset = queryset.filter(update_time__gte=start_time)

        if end_time is not None:
            end_time = parse_datetime(end_time)
            if end_time:
                queryset = queryset.filter(update_time__lte=end_time)

        return queryset
    


class DetailReport(generics.RetrieveUpdateDestroyAPIView):
    queryset = Report.objects.all()
    serializer_class = ReportSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'id'


class ShowXRayImageView(APIView):
    permission_classes = [AllowAny]
    def get(self,request,file_name):
        image_file = X_ray_folder + file_name
        return FileResponse(open(image_file, 'rb'))

class ListAnalysisView(generics.ListAPIView):
    serializer_class = ReportAnalysisSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPagination
    def get_queryset(self):
        queryset = Report.objects.select_related('patient').order_by('-update_time')        
        # 获取查询参数
        patient = self.request.query_params.get('patient', None)
        start_time = self.request.query_params.get('start_time', None)
        end_time = self.request.query_params.get('end_time', None)
        
        # age
        min_age = self.request.query_params.get('min_age', None)
        max_age = self.request.query_params.get('max_age', None)
        age = self.request.query_params.get('age', None)
        # region
        region = self.request.query_params.get('region', None)
        # gender
        gender = self.request.query_params.get('gender', None)
        result = self.request.query_params.get('result', None)
        if result is not None:
            queryset = queryset.filter(result=result)
        if patient is not None:
            queryset = queryset.filter(patient=patient)

        # 解析开始时间和结束时间
        if start_time is not None:
            start_time = parse_datetime(start_time)
            if start_time:
                queryset = queryset.filter(update_time__gte=start_time)

        if end_time is not None:
            end_time = parse_datetime(end_time)
            if end_time:
                queryset = queryset.filter(update_time__lte=end_time)

        # 过滤    
        if min_age is not None:
            queryset = queryset.filter(patient__age__gte=min_age)
        if max_age is not None:
            queryset = queryset.filter(patient__age__lte=max_age)
        if region is not None:
            queryset = queryset.filter(patient__region=region)
        if gender is not None:
            queryset = queryset.filter(patient__gender=gender)
        if age is not None:
            queryset = queryset.filter(patient__age=age)
        return queryset

# class DownLoadZipView(APIView):
#     permission_classes = [IsAuthenticated]
#     def get(self,request,file_name):
#         return FileResponse(open("3.zip", 'rb'))


# def show_xray_image(request,file_name):
#     print(file_name)
#     image_file = X_ray_folder + file_name
#     print(image_file)
#     return FileResponse(open(image_file, 'rb'))