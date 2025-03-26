from django.urls import include, path

app_name = 'api'

urlpatterns = [
    path('multilenguage/', include('EchoTribe.api.endpoints', namespace='multilenguage')),
]
