from django.urls import path
from .views.MultilenguageView import predecir_sentimiento

app_name = "EchoTribe"

urlpatterns = [
    path("predecir/", predecir_sentimiento, name="predecir"),
]