from django.contrib import admin
from django.urls import path, include

from ai.views import RecommendedActivityAPIView, RecommendedAClubAPIView, ReportReplyAPI
from club.views import ClubIntroView, ClubCreateView, ClubDetailView, ClubMemberAPI, \
    ClubNoticeAPI, ClubOngoingActivityAPI, ClubFinishedActivityAPI, ClubPrPostWriteView, ClubPrPostDetailView, \
    ClubTeenplayAPIView, ClubTeenplayDeleteAPIView, ClubTeenplayUploadAPIView, ClubPrPostListView, ClubPrPostReplyAPI, \
    ClubPrPostUpdateView, ClubPrPostDeleteView, ClubPrPostListAPI

app_name = 'ai'

urlpatterns = [
    # 모임 소개 페이지 이동 주소
    # path('intro/', ClubIntroView.as_view(), name='intro'),
    path('activities/', RecommendedActivityAPIView.as_view(), name='recommended-activities'),
    path('clubs/', RecommendedAClubAPIView.as_view(), name='recommended-clubs'),
    path('report/', ReportReplyAPI.as_view(), name='report'),
]