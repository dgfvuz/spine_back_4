from django.urls import path
from .views import CreateReport, ListReport, DetailReport, RegenerateReportView

urlpatterns = [
    path('create/', CreateReport.as_view(), name='create_report'),
    path('list/', ListReport.as_view(), name='list_report'),
    path('detail/<int:id>/', DetailReport.as_view(), name='detail_report'),
    path('regenerate/<int:id>/', RegenerateReportView.as_view(), name='regenerate_report'),
]