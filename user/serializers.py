from rest_framework import serializers
from .models import CustomUser

class CustomUserSerializer(serializers.ModelSerializer):
    avatar = serializers.ImageField(max_length=None, use_url=True, required=False)
    class Meta:
        model = CustomUser
        fields = ('account','password','name','email','description','is_active','is_staff','is_superuser','avatar')
        # 设置名称为非必须
        extra_kwargs = {'name': {'required': False}}
        extra_kwargs = {'password': {'write_only': True}}
    
    def create(self, validated_data):
        user = CustomUser.objects.create_user(**validated_data)
        user.set_password(validated_data['password'])
        user.save()
        return user
    
    def update(self, instance, validated_data):
        if 'name' in validated_data:
            instance.name = validated_data.get('name', instance.name)
        if 'email' in validated_data:
            instance.email = validated_data.get('email', instance.email)
        if 'description' in validated_data:
            instance.description = validated_data.get('description', instance.description)
        if 'is_active' in validated_data:
            instance.is_active = validated_data.get('is_active', instance.is_active)
        if 'is_staff' in validated_data:
            instance.is_staff = validated_data.get('is_staff', instance.is_staff)
        if 'is_superuser' in validated_data:
            instance.is_superuser = validated_data.get('is_superuser', instance.is_superuser)
        if 'password' in validated_data:
            instance.set_password(validated_data['password'])
        if 'avatar' in validated_data:
            instance.avatar = validated_data.get('avatar', instance.avatar)
        instance.save()
        return instance