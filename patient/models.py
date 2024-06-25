from django.db import models

# Create your models here.

class Patient(models.Model):
    # 患者号
    id = models.AutoField(primary_key=True)
    # 患者姓名
    name = models.CharField(max_length=50)
    # 患者年龄
    age = models.IntegerField()
    # 患者地区
    region = models.CharField(max_length=100,blank=True)
    # 患者地址
    address = models.CharField(max_length=100,blank=True)
    # 患者电话
    phone = models.CharField(max_length=20,blank=True)
    # 患者身份证号
    id_card = models.CharField(max_length=20)
    # 患者性别
    gender = models.CharField(max_length=10)
    # 患者病史
    medical_history = models.TextField(blank=True)
    # 患者过敏史
    allergy_history = models.TextField(blank=True)
    # 患者创建时间
    create_time = models.DateTimeField(auto_now_add=True)
    # 患者更新时间
    update_time = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

