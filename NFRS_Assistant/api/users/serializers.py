from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile, ApiKey

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'is_staff']
        read_only_fields = ['id', 'is_staff']


class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = UserProfile
        fields = ['user', 'preferred_language', 'organization', 'is_admin', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']


class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    confirm_password = serializers.CharField(write_only=True)
    preferred_language = serializers.CharField(required=False, default='en')
    organization = serializers.CharField(required=False, allow_blank=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'confirm_password', 'first_name', 'last_name', 'preferred_language', 'organization']

    def validate(self, data):
        if data['password'] != data.pop('confirm_password'):
            raise serializers.ValidationError("Passwords do not match")
        return data

    def create(self, validated_data):
        # Extract profile data before creating user
        preferred_language = validated_data.pop('preferred_language', 'en')
        organization = validated_data.pop('organization', None)

        # Create the user
        user = User.objects.create_user(**validated_data)

        # Update the profile created by the signal instead of creating a new one
        try:
            profile = UserProfile.objects.get(user=user)
            profile.preferred_language = preferred_language
            if organization:
                profile.organization = organization
            profile.save()
        except UserProfile.DoesNotExist:
            # This shouldn't happen because of the signal, but just in case
            UserProfile.objects.create(
                user=user,
                preferred_language=preferred_language,
                organization=organization
            )

        return user


class ApiKeySerializer(serializers.ModelSerializer):
    class Meta:
        model = ApiKey
        fields = ['id', 'key', 'name', 'is_active', 'created_at', 'last_used_at']
        read_only_fields = ['id', 'key', 'created_at', 'last_used_at']


class ChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True)
    confirm_password = serializers.CharField(required=True)

    def validate(self, data):
        if data['new_password'] != data['confirm_password']:
            raise serializers.ValidationError("New passwords do not match")
        return data