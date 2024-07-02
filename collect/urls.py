from django.urls import path
from .views import CollectCreateView, CollectListView, CollectDetailView

urlpatterns = [
    path('create/', CollectCreateView.as_view(), name='create_collect'),
    path('list/', CollectListView.as_view(), name='list_collect'),
    path('detail/<int:id>/', CollectDetailView.as_view(), name='detail_collect'),
]
