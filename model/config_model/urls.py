from django.urls import path,include

from .views import home, register, login_user, main_page, predict_quality, rag_usage

urlpatterns = [
    path("1",home),
    path("register/",register,name="register"),
    path("login/",login_user,name="login"),
    path("main/",main_page, name="main"),
    path("prediction/",predict_quality,name="prediction"),
    path("rag/",rag_usage,name="rag")


]