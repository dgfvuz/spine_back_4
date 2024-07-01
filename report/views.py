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
from .serializers import ReportSerializer, GenerateReportSerializer
from patient.pagination import CustomPagination
from .config import X_ray_folder
from .model import getResult
from django.http import HttpResponse
from rest_framework import status
from rest_framework.views import APIView

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
    results = getResult(instance.X_ray)
    instance.results = results
    instance.status = '未审核'
    instance.save()    

# 定义HTTP方法为post
class RegenerateReportView(generics.GenericAPIView):
    serializer_class = GenerateReportSerializer
    permission_classes = [IsAuthenticated]

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
        queryset = Report.objects.all().order_by('-update_time')
        # 获取查询参数
        patient_id = self.request.query_params.get('patient_id', None)
        if patient_id is not None:
            queryset = queryset.filter(patient_id=patient_id)
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


# class DownLoadZipView(APIView):
#     permission_classes = [IsAuthenticated]
#     def get(self,request,file_name):
#         return FileResponse(open("3.zip", 'rb'))


# def show_xray_image(request,file_name):
#     print(file_name)
#     image_file = X_ray_folder + file_name
#     print(image_file)
#     return FileResponse(open(image_file, 'rb'))