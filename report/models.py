from django.db import models
from patient.models import Patient
from .config import X_ray_folder

# Create your models here.
class Report(models.Model):
    id = models.AutoField(primary_key=True)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE,related_name='reports')
    X_ray = models.ImageField(upload_to=X_ray_folder,blank=True)
    # 结果
    results = models.JSONField(blank=True, null=True)  # 允许字段为空或者为null，适用于可选字段
    result = models.CharField(max_length=20, default='正常')
    # 创建时间
    # status有各种状态：生成中、生成失败、未审核、已审核
    status = models.CharField(max_length=20, default='生成中')
    create_time = models.DateTimeField(auto_now_add=True)
    # 更新时间
    update_time = models.DateTimeField(auto_now=True)