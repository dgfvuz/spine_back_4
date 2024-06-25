from django.urls import path
from .views import CreatePatient, ListPatient, DetailPatient

urlpatterns = [
    path('create/', CreatePatient.as_view(), name='create_patient'),
    path('list/', ListPatient.as_view(), name='list_patient'),
    path('detail/<int:id>/', DetailPatient.as_view(), name='detail_patient'),
]