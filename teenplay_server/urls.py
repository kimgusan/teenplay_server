"""
URL configuration for teenplay_server project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

from main.views import MainView, FooterNoticeLatestAPI
from teenplay_server.views import AdminLoginView, AdminUserView, CompanyIntroductionView, CompanyNoticeListAPI, \
    AdminMessageView, AdminPromoteView, AdminActivityView, AdminWishlistView, AdminMeetingView, \
    AdminFestivalView, AdminNoticeView, AdminNoticeWriteView, AdminCommentView, AdminUserUpdateAPI, \
    AdminNoticePaginationAPI, AdminWishlistAPI, AdminNoticeUpdateAPI, AdminUserAPI, \
    AdminCommentAPI, AdminCommentDeleteAPI, AdminActivityAPI, AdminActivityDeleteAPI, AdminPromoteAPI, \
    AdminPromoteDeleteAPI, AdminMessageAPI, \
    AdminWishlistDeleteAPI, AdminMeetingAPI, AdminMeetingDeleteAPI, AdminFestivalPaginationAPI, AdminFestivalUpdateAPI, \
    AdminFestivalWriteView, AdminLogoutView


# AdminTeenplayView, AdminTeenplayAPI, AdminTeenplayDeleteAPI


# urls에 음수 값 넣기 가능!
class IntConverter:
    regex = '-?\d+'

    def to_python(self, value):
        return int(value)

    def to_url(self, value):
        return str(value)


urlpatterns = [
    # ai 서비스
    path('ai/api/', include('ai.urls')),
    # path('admin/', admin.site.urls),
    path('club/', include('club.urls-web')),
    path('clubs/', include('club.urls-web')),
    path('app/club/', include('club.urls-app')),
    path('app/clubs/', include('club.urls-app')),
    path('member/', include('member.urls-web')),
    path('app/member/', include('member.urls-app')),
    path('accounts/', include('allauth.urls')),
    path('oauth/', include('oauth.urls')),
    path('teenplay/', include('teenplay.urls-web')),
    path('app/teenplay/', include('teenplay.urls-app')),
    path('notice/', include('notice.urls-web')),
    path('notice/', include('notice.urls-web')),
    path('app/notice/', include('notice.urls-app')),
    path('festival/', include('festival.urls-web')),
    path('app/festival/', include('festival.urls-app')),
    # 관리자
    path('admin/login/', AdminLoginView.as_view(), name='admin-login'),
    path('admin/logout/', AdminLogoutView.as_view(), name='admin-login'),
    # 관리자 - 유저 관리
    path('admin/user/', AdminUserView.as_view(), name='admin-user'),
    path('admin/users/<int:page>/', AdminUserAPI.as_view(), name='admin-user-api'),
    path('admin/user/update/<int:member_id>/', AdminUserUpdateAPI.as_view(), name='admin-user-update'),
    # 관리자 - 쪽지 관리
    path('admin/message/', AdminMessageView.as_view(), name='admin-message'),
    path('admin/messages/<int:page>/', AdminMessageAPI.as_view(), name='admin-message-api'),
    # 관리자 - 틴플레이 관리
    # path('admin/teenplay/', AdminTeenplayView.as_view(), name='admin-teenplay'),
    # path('admin/teenplaies/<int:page>/', AdminTeenplayAPI.as_view(), name='admin-teenplay-api'),
    # path('admin/teenplaies/delete/<int:teenplay_id>/', AdminTeenplayDeleteAPI.as_view(), name='admin-teenplay-delete'),
    # 관리자 - 게시글 홍보글 관리
    path('admin/promote/', AdminPromoteView.as_view(), name='admin-promote'),
    path('admin/promotes/<int:page>/', AdminPromoteAPI.as_view(), name='admin-promote-api'),
    path('admin/promotes/delete/<int:promote_id>/', AdminPromoteDeleteAPI.as_view(), name='admin-promote-delete'),
    # 관리자 - 게시글 활동모집글 관리
    path('admin/activity/', AdminActivityView.as_view(), name='admin-activity'),
    path('admin/activities/<int:page>/', AdminActivityAPI.as_view(), name='admin-activity-api'),
    path('admin/activities/delete/<int:activity_id>/', AdminActivityDeleteAPI.as_view(), name='admin-activity-delete'),
    # 관리자 - 게시글 위시리스트 관리
    path('admin/wishlist/', AdminWishlistView.as_view(), name='admin-wishlist'),
    path('admin/wishlists/<int:page>/', AdminWishlistAPI.as_view(), name='admin-wishlist-api'),
    path('admin/wishlists/delete/<int:wishlist_id>/', AdminWishlistDeleteAPI.as_view(), name='admin-wishlist-delete'),
    # 관리자 - 전체 모임 관리
    path('admin/meeting/', AdminMeetingView.as_view(), name='admin-meeting'),
    path('admin/meetings/<int:page>/', AdminMeetingAPI.as_view(), name='admin-meeting-api'),
    path('admin/meetings/delete/<int:meeting_id>', AdminMeetingDeleteAPI.as_view(), name='admin-meeting-delete'),
    # 관리자 - 축제 관리
    path('admin/festival/', AdminFestivalView.as_view(), name='admin-festival'),
    path('admin/festivals/<int:page>/', AdminFestivalPaginationAPI.as_view(), name='admin-festival-page-api'),
    path('admin/festivals/delete/<int:festival_id>/', AdminFestivalUpdateAPI.as_view(), name='admin-festival-delete-api'),
    path('admin/festival/write/', AdminFestivalWriteView.as_view(), name='admin-festival-write'),
    # 관리자 - 공지사항 관리
    path('admin/notice/', AdminNoticeView.as_view(), name='admin-notice'),
    path('admin/notices/<int:page>/', AdminNoticePaginationAPI.as_view(), name='admin-notice-page-api'),
    path('admin/notices/update/<int:notice_id>/', AdminNoticeUpdateAPI.as_view(), name='admin-notice-delete-api'),
    path('admin/notice/write/', AdminNoticeWriteView.as_view(), name='admin-notice-write'),
    # 관리자 - 댓글 관리
    path('admin/comment/', AdminCommentView.as_view(), name='admin-comment'),
    path('admin/comments/<int:page>/', AdminCommentAPI.as_view(), name='admin-comment-api'),
    path('admin/comments/delete/', AdminCommentDeleteAPI.as_view(), name='admin-comment-delete'),
    # 여기까지~
    path('terms/', include('terms.urls-web')),
    path('app/terms/', include('terms.urls-app')),
    path('activity/', include('activity.urls-web')),
    path('app/activity/', include('activity.urls-app')),
    path('company/', CompanyIntroductionView.as_view(), name='company'),
    path('app/company/', CompanyIntroductionView.as_view(), name='company'),
    path('company/notice/api/<int:page>/', CompanyNoticeListAPI.as_view(), name='company-api'),
    path('footer/notice/api/', FooterNoticeLatestAPI.as_view(), name='footer-notice-api'),
    path('pay/', include('pay.urls')),
    path('wishlist/', include('wishlist.urls')),
    path('search/', include('main.urls-web')),
    path('app/search/', include('main.urls-app')),
    path('', MainView.as_view())
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
