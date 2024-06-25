from rest_framework import serializers
from .models import Patient

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = '__all__'

class PatientCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = ('name','age','region','address','phone','id_card','gender','medical_history','allergy_history')
        # 设置名称为非必须
        extra_kwargs = {'name': {'required': False}}
        extra_kwargs = {'age': {'required': False}}



    
