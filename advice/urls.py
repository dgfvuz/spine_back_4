from django.urls import path
from .views import AdviceCreateView, AdviceListView

urlpatterns = [
    path('create/', AdviceCreateView.as_view(), name='create_advice'),
    path('list/', AdviceListView.as_view(), name='list_advice'),
]