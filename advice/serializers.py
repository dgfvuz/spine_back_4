from rest_framework import serializers
from .models import Advice

class AdviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Advice
        fields = '__all__'




    
