from rest_framework import serializers
from .models import Report
from rest_framework.fields import SerializerMethodField

class ReportSerializer(serializers.ModelSerializer):
    patient_name = SerializerMethodField()  # 声明SerializerMethodField
    class Meta:
        model = Report
        fields = ['id', 'patient', 'X_ray', 'results', 'status', 'create_time','update_time','patient_name']

    def get_patient_name(self, obj):
        return obj.patient.name

class GenerateReportSerializer(serializers.Serializer):
    class Meta:
        model = Report
        fields = {'id'}