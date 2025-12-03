from django.core.validators import RegexValidator
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _
# Create your models here.

phone_validator = RegexValidator(
    regex=r'^\+?1?\d{9,15}$',
    message="Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed."
)


class CustomUser(AbstractUser):
    email = models.EmailField(_("email address"), blank=False, unique=True)
    username = models.CharField(max_length=10,blank=True,null=True,unique=True)
    first_name = models.CharField(max_length=10,blank=True)
    last_name = models.CharField(max_length=20,blank=True)
    phone_number = models.CharField(validators=[phone_validator], max_length=17, blank=True,
                                    verbose_name="Short_biography")

