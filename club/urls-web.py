from django.contrib import admin
from django.urls import path, include

from club.views import ClubIntroView, ClubCreateView, ClubDetailView, ClubMemberAPI, \
    ClubNoticeAPI, ClubOngoingActivityAPI, ClubFinishedActivityAPI, ClubPrPostWriteView, ClubPrPostDetailView, \
    ClubTeenplayAPIView, ClubTeenplayDeleteAPIView, ClubTeenplayUploadAPIView, ClubPrPostListView, ClubPrPostReplyAPI, \
    ClubPrPostUpdateView, ClubPrPostDeleteView, ClubPrPostListAPI

app_name = 'club'

urlpatterns = [
    # 모임 소개 페이지 이동 주소
    path('intro/', ClubIntroView.as_view(), name='intro'),
    # 모임 개설 페이지 이동 주소
    path('create/', ClubCreateView.as_view(), name='create'),
    # 모임 상세 페이지 이동 주소
    path('detail/', ClubDetailView.as_view(), name='detail'),
    path('club-member/api/<int:member_id>/<int:club_id>/', ClubMemberAPI.as_view(), name='club-member-api'),
    path('club-ongoing-activity/api/<int:club_id>/', ClubOngoingActivityAPI.as_view(), name='club-ongoing-activity-api'),
    path('club-finished-activity/api/<int:club_id>/<int:page>/', ClubFinishedActivityAPI.as_view(), name='club-finished-activity-api'),
    path('club-notice/api/<int:club_id>/<int:page>/', ClubNoticeAPI.as_view(), name='club-notice-api'),
    path('club-teenplay-list/api/<int:club_id>/<int:page>/', ClubTeenplayAPIView.as_view(), name='club-teenplay-list'),
    path('club-teenplay-delete/api/<int:teenplay_id>/', ClubTeenplayDeleteAPIView.as_view(),name='club-teenplay-delete'),
    path('club-teenplay-upload/api/', ClubTeenplayUploadAPIView.as_view(), name='club-teenplay-upload'),
    # 모임 홍보글 목록 페이지 이동 주소
    path('pr-post-list/', ClubPrPostListView.as_view(), name='pr-post-list'),
    path('pr-post-list/api/', ClubPrPostListAPI.as_view(), name='pr-post-list-api'),
    # 모임 홍보글 상세 페이지 이동 주소
    path('pr-post-detail/', ClubPrPostDetailView.as_view(), name='pr-post-detail'),
    path('pr-post-reply/api/', ClubPrPostReplyAPI.as_view(), name='pr-post-reply'),
    path('pr-post-delete/', ClubPrPostDeleteView.as_view(), name='pr-post-delete'),
    # 모임 홍보글 작성 페이지 이동 주소
    path('pr-post-write/', ClubPrPostWriteView.as_view(), name='pr-post-write'),
    # 모임 홍보글 수정 페이지 이동 주소
    path('pr-post-update/', ClubPrPostUpdateView.as_view(), name='pr-post-update'),
]
