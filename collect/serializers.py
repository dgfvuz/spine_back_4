from .models import Collect
from rest_framework import serializers
from rest_framework.fields import SerializerMethodField

class CollectSerializer(serializers.ModelSerializer):
    patient_name = SerializerMethodField()  # 声明SerializerMethodField
    patient_age = SerializerMethodField()  # 声明SerializerMethodField
    patient_region = SerializerMethodField()  # 声明SerializerMethodField
    patient_address = SerializerMethodField()  # 声明SerializerMethodField
    patient_phone = SerializerMethodField()  # 声明SerializerMethodField
    patient_id_card = SerializerMethodField()  # 声明SerializerMethodField
    patient_gender =   SerializerMethodField()  # 声明SerializerMethodField
    patient_medical_history = SerializerMethodField()  # 声明SerializerMethodField
    patient_allergy_history = SerializerMethodField()  # 声明SerializerMethodField
    patient_create_time = SerializerMethodField()  # 声明SerializerMethodField
    patient_update_time = SerializerMethodField()  # 声明SerializerMethodField
    class Meta:
        model = Collect
        fields = ['id', 'patient', 'user', 'timestamp','patient_name','patient_age','patient_region','patient_address','patient_phone','patient_id_card','patient_gender','patient_medical_history','patient_allergy_history','patient_create_time','patient_update_time']
    def get_patient_name(self, obj):
        return obj.patient.name
    def get_patient_age(self, obj):
        return obj.patient.age
    def get_patient_region(self, obj):
        return obj.patient.region
    def get_patient_address(self, obj):
        return obj.patient.address
    def get_patient_phone(self, obj):
        return obj.patient.phone
    def get_patient_id_card(self, obj):
        return obj.patient.id_card
    def get_patient_gender(self, obj):
        return obj.patient.gender
    def get_patient_medical_history(self, obj):
        return obj.patient.medical_history
    def get_patient_allergy_history(self, obj):
        return obj.patient.allergy_history
    def get_patient_create_time(self, obj):
        return obj.patient.create_time
    def get_patient_update_time(self, obj):
        return obj.patient.update_time
