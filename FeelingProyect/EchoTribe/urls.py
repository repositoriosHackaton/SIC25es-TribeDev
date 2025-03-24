from django.urls import path
from . import views
app = "EchoTribe"

urlpatterns = [
    path('evaluar', views.evaluar_texto, name="evaluar")
]