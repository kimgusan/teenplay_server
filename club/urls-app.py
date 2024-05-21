from django.contrib import admin
from django.urls import path, include

from club.views import ClubIntroView, ClubCreateView, ClubDetailView, ClubPrPostListView, \
    ClubPrPostDetailView, ClubPrPostWriteView

app_name = 'app-club'

urlpatterns = [
    path('intro/', ClubIntroView.as_view(), name='intro'),
    path('create/', ClubCreateView.as_view(), name='create'),
    path('detail/', ClubDetailView.as_view(), name='detail'),
    path('pr-post-list/', ClubPrPostListView.as_view(), name='pr-post-list'),
    path('pr-post-detail/', ClubPrPostDetailView.as_view(), name='pr-post-detail'),
    path('pr-post-write/', ClubPrPostWriteView.as_view(), name='pr-post-write'),
    path('pr-post-update/', ClubPrPostWriteView.as_view(), name='pr-post-update'),
]
