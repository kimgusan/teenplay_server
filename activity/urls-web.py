from django.contrib import admin
from django.urls import path, include

from activity.views import ActivityCreateWebView, ActivityDetailWebView

app_name = 'activity'

urlpatterns = [
    path('create/', ActivityCreateWebView.as_view(), name='create'),
    path('detail/', ActivityDetailWebView.as_view(), name='detail'),
]