from rest_framework import serializers
from .models import Patient

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = '__all__'


class PatientRegionSerializer(serializers.Serializer):
    region = serializers.CharField(max_length=100)

    
