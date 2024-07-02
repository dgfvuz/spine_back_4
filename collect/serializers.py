from .models import Collect
from rest_framework import serializers

class CollectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Collect
        fields = '__all__'