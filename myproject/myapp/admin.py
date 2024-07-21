from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .forms import CustomUserCreationForm, CustomUserChangeForm
from .models import CustomUser

class CustomUserAdmin(UserAdmin):
    add_form = CustomUserCreationForm
    form = CustomUserChangeForm
    model = CustomUser
    list_display = ['username', 'email', 'first_name', 'user_id', 'type', 'is_staff', 'is_active']
    
    #Adds additional fields to display in the admin list view
    fieldsets = UserAdmin.fieldsets + (
        (None, {'fields': ('user_id', 'type')}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        (None, {'fields': ('user_id', 'type')})
    )
#Registers the custom user model with the custom admin class
admin.site.register(CustomUser, CustomUserAdmin)