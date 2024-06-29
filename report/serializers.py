from rest_framework import serializers
from .models import Report

class ReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Report
        fields = '__all__'

class GenerateReportSerializer(serializers.Serializer):
    class Meta:
        model = Report
        fields = {'id'}