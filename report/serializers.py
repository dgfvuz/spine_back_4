from rest_framework import serializers
from .models import Report
from rest_framework.fields import SerializerMethodField

class ReportSerializer(serializers.ModelSerializer):
    patient_name = SerializerMethodField()  # 声明SerializerMethodField
    class Meta:
        model = Report
        fields = ['id', 'patient', 'X_ray', 'results', 'status', 'create_time','update_time','patient_name','result']

    def get_patient_name(self, obj):
        return obj.patient.name

class GenerateReportSerializer(serializers.Serializer):
    class Meta:
        model = Report
        fields = {'id'}

class ReportAnalysisSerializer(serializers.Serializer):
    patient_age = SerializerMethodField()
    patient_region = SerializerMethodField()
    patient_gender = SerializerMethodField()
    class Meta:
        model = Report
        fields = {'id','patient','patient_age','patient_region','patient_gender','result','update_time'}
    
    def get_patient_age(self, obj):
        return obj.patient.age
    
    def get_patient_region(self, obj):
        return obj.patient.region
    
    def get_patient_gender(self, obj):
        return obj.patient.gender
    