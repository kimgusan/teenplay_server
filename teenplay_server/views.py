import math

from django.db import transaction
from django.db.models import Q, Count, F
from django.shortcuts import render, redirect
from django.views import View
from rest_framework.response import Response
from rest_framework.views import APIView

from club.models import Club
from member.models import AdminAccount, Member
from member.serializers import AdminAccountSerializer
from notice.models import Notice

from django.utils import timezone


# 관리자 로그인 페이지 이동
class AdminLoginView(View):
    def get(self, request):
        return render(request, 'admin/web/admin-login-web.html')

    def post(self, request):
        data = request.POST

        data = {
            'admin_id': data['admin_id'],
            'admin_password': data['admin_password']
        }

        admin = AdminAccount.objects.filter(**data)

        context = {'admin_data': None}

        if admin.exists():
            admin_data = AdminAccountSerializer(admin.first()).data
            context['admin_data'] = admin_data

        return render(request, 'admin/web/user-web.html', context)

        # url = 'admin/web/admin-login-web.html'
        #
        # if admin.exists():
        #     request.session['admin'] = AdminAccountSerializer(admin.first()).data
        #
        #     url = 'admin/web/user-web.html'
        #
        # return redirect(url)


# 추가
# 관리자 유저 관리 페이지로 이동
# class AdminUserView(View):
#     def get(self, request):
#         return render(request, 'admin/web/user-web.html')


# # 관리자 유저 관리 페이지 - 유저 정보 불러오기
# class AdminUserAPI(APIView):
#     def get(self, request, page):
#         row_count = 5
#
#         offset = (page - 1) * row_count
#         limit = page * row_count
#
#         users = Member.objects.filter(Q(status=1) | Q(status=2))\
#                     .annotate(club_count=Count('club'), club_action_count=Count('clubmember', filter=Q(status=1)), activity_count=Count('club__activity'))\
#                     .values('member_nickname', 'created_date', 'club_count', 'club_action_count', 'activity_count', 'status')[offset:limit]
#
#         has_next = Member.objects.filter(Q(status=1) | Q(status=2))[limit:limit + 1].exists()
#
#         user_info = {
#             'users': users,
#             'hasNext': has_next,
#         }
#
#         return Response(user_info)


class AdminUserView(View):
    def get(self, request):
        order = request.GET.get('order', 'recent')
        page = int(request.GET.get('page', 1))

        row_count = 5

        offset = (page - 1) * row_count
        limit = page * row_count

        total = Member.objects.filter(Q(status=1) | Q(status=2)).all().count()

        page_count = 5

        end_page = math.ceil(page / page_count) * page_count
        start_page = end_page - page_count + 1
        real_end = math.ceil(total / row_count)
        end_page = real_end if end_page > real_end else end_page

        if end_page == 0:
            end_page = 1

        context = {
            'order': order,
            'start_page': start_page,
            'end_page': end_page,
            'page': page,
            'real_end': real_end,
            'page_count': page_count,
        }
        ordering = '-id'
        if order == 'popular':
            ordering = '-post_read_count'

        context['users'] = list(Member.objects.filter(Q(status=1) | Q(status=2))\
                    .annotate(club_count=Count('club'), club_action_count=Count('clubmember__id', filter=Q(clubmember__status=1)), activity_count=Count('club__activity'))\
                    .values('member_nickname', 'created_date', 'club_count', 'club_action_count', 'activity_count', 'status').order_by(ordering))[offset:limit]
        print(context['users'])

        return render(request, 'admin/web/user-web.html', context)


class AdminUserUpdateView(View):
    @transaction.atomic
    def get(self, request):
        user = Member.objects.get(id=request.GET['id'])
        user.status = -1
        user.updated_date = timezone.now()
        user.save(update_fields=['status', 'updated_date'])

        return redirect('admin/web/user-web.html')


# 관리자 쪽지 관리 페이지 이동
class AdminMessageView(View):
    def get(self, request):
        return render(request, 'admin/web/message-web.html')


# 관리자 틴 플레이 관리 페이지 이동
class AdminTeenplayView(View):
    def get(self, request):
        return render(request, 'admin/web/teenplay-web.html')


# 관리자 게시글 - 홍보글 관리 페이지 이동
class AdminPromoteView(View):
    def get(self, request):
        return render(request, 'admin/web/promote-web.html')


# 관리자 게시글 - 활동 모집글 관리 페이지 이동
class AdminActivityView(View):
    def get(self, request):
        return render(request, 'admin/web/activity-web.html')


# 관리자 게시글 - 위시리스트 관리 페이지 이동
class AdminWishlistView(View):
    def get(self, request):
        return render(request, 'admin/web/wishlist-web.html')


# 관리자 전체 모임 관리 페이지 이동
class AdminMeetingView(View):
    def get(self, request):
        return render(request, 'admin/web/meeting-web.html')


# 관리자 축제 관리 페이지 이동
class AdminFestivalView(View):
    def get(self, request):
        return render(request, 'admin/web/festival-list-web.html')


# 관리자 축제 작성 페이지 이동
class AdminFestivalWrite(View):
    def get(self, request):
        return render(request, 'admin/web/festival-create-web.html')


# 관리자 공지사항 관리 페이지 이동
class AdminNoticeView(View):
    def get(self, request):
        return render(request, 'admin/web/notice-list-web.html')


# 관리자 공지사항 작성 페이지 이동
class AdminNoticeWriteView(View):
    def get(self, request):
        return render(request, 'admin/web/notice-create-web.html')


# 관리자 댓글 관리 페이지 이동
class AdminCommentView(View):
    def get(self, request):
        return render(request, 'admin/web/comment-web.html')
# 여기까지~


# 회사 소개 페이지 이동
class CompanyIntroductionView(View):
    def get(self, request):
        return render(request, 'company-info/web/company-info-web.html')


# 회사 소개에서 공지사항 띄우기
class CompanyNoticeListAPI(APIView):
    def get(self, request, page):
        row_count = 7
        offset = (page - 1) * row_count
        limit = page * row_count

        notices = Notice.objects.filter(status=True).values()[offset:limit]

        has_next = Notice.objects.filter(status=True)[limit:limit+1].exists()

        total = Notice.objects.filter(status=True).count()

        notice_info = {
            'notices': notices,
            'hasNext': has_next,
            'total': total
        }

        return Response(notice_info)

