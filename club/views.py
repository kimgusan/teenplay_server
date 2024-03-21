import math
import os.path

from django.db import transaction, models
from django.db.models import Count, F, Q, OuterRef, Subquery, Count
from django.db.models.functions import Coalesce, Concat
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views import View
from rest_framework.response import Response
from rest_framework.views import APIView

from activity.models import Activity, ActivityLike
from alarm.models import Alarm
from club.models import Club, ClubMember, ClubPost, ClubPostReply
from member.models import Member
from teenplay.models import TeenPlay
from teenplay_server import settings
from teenplay_server.category import Category


class ClubIntroView(View):
    def get(self, request):
        return render(request, 'club/web/club-intro-web.html')


class ClubCreateView(View):
    def get(self, request):
        return render(request, 'club/web/club-create-web.html')

    @transaction.atomic
    def post(self, request):
        data = request.POST
        file = request.FILES

        member = Member(**request.session['member'])

        data = {
            'club_name': data['club-name'],
            'club_intro': data['club-intro'],
            'member': member,
            'club_profile_path': file.get('club-profile'),
            'club_banner_path': file.get('club-banner')
        }

        club = Club.objects.create(**data)

        return redirect(club.get_absolute_url())


class ClubDetailView(View):
    def get(self, request):
        club_id = request.GET['id']
        view = request.GET.get('view', 'activity')

        columns = [
            'id',
            'club_name',
            'club_intro',
            'club_info',
            'club_profile_path',
            'club_banner_path',
            'owner_id',
            'owner_name',
            'owner_email',
            'owner_phone',
        ]

        club_list = Club.objects.filter(id=club_id) \
            .annotate(
            owner_id=F('member__id'),
            owner_name=F('member__member_nickname'),
            owner_email=F('member__member_email'),
            owner_phone=F('member__member_phone')).values(*columns) \
            .annotate(club_member_count=Count('clubmember', filter=Q(clubmember__status=1)))

        club_activity_count = Club.objects.filter(id=club_id).values('id') \
            .annotate(club_activity_count=Count('activity')).first()

        club_list = list(club_list)

        club_list[0]['club_activity_count'] = club_activity_count.get('club_activity_count')
        club_list[0]['view'] = view

        if club_list[0]['club_info'] is None:
            club_list[0]['club_info'] = ''

        context = {
            'club_list': club_list
        }

        return render(request, 'club/web/club-detail-web.html', context)


class ClubAPI(APIView):
    def get(self, request, club_id):
        club = Club.objects.filter(id=club_id).values().first()

        return Response(club)


class ClubMemberAPI(APIView):
    def get(self, request, member_id, club_id):
        club_members = ClubMember.objects.filter(member=member_id, club=club_id).values()

        return Response(club_members)

    @transaction.atomic
    def patch(self, request, member_id, club_id):
        # data = request.data
        member = Member.objects.get(id=member_id)
        club = Club.objects.get(id=club_id)
        club_member, created = ClubMember.objects.get_or_create(member=member, club=club)

        message = 'create-apply'
        flag = True
        alarm_type = 9

        if not created:
            if club_member.status == -1:
                club_member.status = 0
                club_member.updated_date = timezone.now()
                club_member.save(update_fields=['status', 'updated_date'])

                message = 'cancel'
                flag = False

            elif club_member.status == 0:
                club_member.status = -1
                club_member.updated_date = timezone.now()
                club_member.save(update_fields=['status', 'updated_date'])

                message = 'apply'

            elif club_member.status == 1:
                club_member.status = 0
                club_member.updated_date = timezone.now()
                club_member.save(update_fields=['status', 'updated_date'])

                message = 'quit'
                alarm_type = 10

        if flag:
            Alarm.objects.create(target_id=club_id, alarm_type=alarm_type, sender=member, receiver=club.member)

        return Response(message)


class ClubOngoingActivityAPI(APIView):
    def get(self, request, club_id):
        member = Member(**request.session['member'])
        club = Club.objects.get(id=club_id)

        ongoing_activities = list(Activity.objects.filter(club=club, activity_end__gt=timezone.now(), status=1)
                                  .values('id', 'activity_title', 'thumbnail_path', 'activity_start',)
                                  .annotate(participant_count=Count('activitymember', filter=Q(activitymember__status=1))))

        for ongoing_activity in ongoing_activities:
            ongoing_activity['is_like'] = ActivityLike.enabled_objects.filter(activity=ongoing_activity['id'],
                                                                              member=member).exists()

        return Response(ongoing_activities)


class ClubFinishedActivityAPI(APIView):
    def get(self, request, club_id, page):
        member = Member(**request.session['member'])

        row_count = 8
        offset = (page - 1) * row_count
        limit = page * row_count

        club = Club.objects.get(id=club_id)

        finished_activities = list(Activity.objects.filter(club=club, activity_end__lte=timezone.now(), status=1)
                                   .values('id', 'activity_title', 'thumbnail_path', 'activity_start')
                                   .annotate(
            participant_count=Count('activitymember', filter=Q(activitymember__status=1)))
                                   .order_by('-id'))

        for finished_activity in finished_activities:
            finished_activity['is_like'] = ActivityLike.enabled_objects.filter(activity=finished_activity['id'],
                                                                               member=member).exists()

        return Response(finished_activities[offset:limit])


class ClubNoticeAPI(APIView):
    def get(self, request, club_id, page):
        row_count = 4
        offset = (page - 1) * row_count
        limit = page * row_count

        club = Club.objects.get(id=club_id)

        club_notices = club.clubnotice_set.filter(status=1).values().order_by('-id')

        return Response(club_notices[offset:limit])


class ClubPrPostWriteView(View):
    def get(self, request):
        club = Club.objects.get(id=request.GET['club_id'])

        context = {
            'club_id': club.id,
            'club_name': club.club_name,
        }

        return render(request, 'club/web/club-pr-posts-write-web.html', context)

        # if club.member_id == request.GET['member']['id']:
        #     context = {
        #         'club_id': club.id,
        #         'club_name': club.club_name,
        #     }
        #
        #     return render(request, 'club/web/club-pr-posts-write-web.html', context)
        # return render(request, '/')

    @transaction.atomic
    def post(self, request):
        datas = request.POST
        files = request.FILES

        category = Category.objects.get(category_name=datas['category'])

        datas = {
            'club': Club.objects.get(id=request.POST['club_id']),
            'post_title': datas['title'],
            'post_content': datas['content'],
            'category': category,
            'image_path': files['image']
        }

        club_post = ClubPost.objects.create(**datas)

        return redirect(club_post.get_absolute_url())


class ClubPrPostDetailView(View):
    def get(self, request):
        club_post = ClubPost.enabled_objects.get(id=request.GET['id'])
        replies = ClubPostReply.enabled_objects.filter(club_post=club_post).values()

        club_post.view_count += 1
        club_post.updated_date = timezone.now()
        club_post.save(update_fields=['view_count', 'updated_date'])

        context = {
            'club_post': club_post,
            'replies': list(replies)
        }

        return render(request, 'club/web/club-pr-posts-detail-web.html', context)


class ClubPrPostUpdateView(View):
    def get(self, request):
        club_post_id = request.GET['id']
        club_post = ClubPost.enabled_objects.get(id=club_post_id)

        file_path = os.path.join(settings.MEDIA_ROOT, club_post.image_path.path)
        file_size = os.path.getsize(file_path)

        context = {
            'club_post': club_post,
            'file_size': file_size
        }

        return render(request, 'club/web/club-pr-posts-update-web.html', context)

    def post(self, request):
        datas = request.POST
        files = request.FILES

        club_post = ClubPost.objects.get(id=datas['club_post_id'])
        category = Category.objects.get(category_name=datas['category'])

        club_post.post_title = datas['title']
        club_post.post_content = datas['content']
        club_post.category = category
        if files:
            club_post.image_delete()
            club_post.image_path = files['image']
            club_post.updated_date = timezone.now()
            club_post.save(update_fields=['post_title', 'post_content', 'category', 'image_path', 'updated_date'])
        else:
            club_post.updated_date = timezone.now()
            club_post.save(update_fields=['post_title', 'post_content', 'category', 'updated_date'])

        return redirect(club_post.get_absolute_url())


class ClubPrPostDeleteView(View):
    def post(self, request):
        datas = request.POST
        club_post = ClubPost.objects.get(id=datas['id'])
        club_post.status = 0
        club_post.updated_date = timezone.now()
        club_post.save(update_fields=['status', 'updated_date'])

        return redirect('/club/pr-post-list/')


class ClubPrPostReplyAPI(APIView):
    def get(self, request):
        page = int(request.GET.get('page', 1))
        club_post_id = request.GET.get('club_post_id')

        row_count = 4
        offset = (page - 1) * row_count
        limit = page * row_count

        replies = ClubPostReply.enabled_objects.filter(club_post_id=club_post_id) \
            .annotate(member_email=F('member__member_email'), member_name=F('member__member_nickname'),
                      member_path=F('member__memberprofile__profile_path')) \
            .values('id', 'reply_content', 'created_date', 'member_id', 'member_email', 'member_name',
                    'member_path').order_by('-id')

        replies_count = replies.count()
        replies_info = {
            'replies': replies[offset:limit],
            'replies_count': replies_count
        }

        return Response(replies_info)

    @transaction.atomic
    def post(self, request):
        data = request.data
        data = {
            'reply_content': data['reply_content'],
            'club_post_id': data['club_post_id'],
            'member_id': request.session['member']['id']
        }

        ClubPostReply.objects.create(**data)

        club_post = ClubPost.enabled_objects.get(id=data['club_post_id'])

        data = {
            'target_id': data['club_post_id'],
            'alarm_type': 1,
            'sender_id': request.session['member']['id'],
            'receiver_id': club_post.club.member_id
        }

        Alarm.objects.create(**data)

        return Response("success")

    @transaction.atomic
    def patch(self, request):
        reply_content = request.data['reply_content']
        reply_id = request.data['id']

        club_post_reply = ClubPostReply.enabled_objects.get(id=reply_id)
        club_post_reply.reply_content = reply_content
        club_post_reply.updated_date = timezone.now()
        club_post_reply.save(update_fields=['reply_content', 'updated_date'])

        return Response("success")

    @transaction.atomic
    def delete(self, request):
        reply_id = request.GET.get('id')

        club_post_reply = ClubPostReply.enabled_objects.get(id=reply_id)
        club_post_reply.status = 0
        club_post_reply.updated_date = timezone.now()
        club_post_reply.save(update_fields=['status', 'updated_date'])

        return Response("success")


class ClubPrPostListView(View):
    def get(self, request):
        keyword = request.GET.get('keyword', '')
        category = request.GET.get('category', '')
        order = request.GET.get('order', '최신순')
        page = request.GET.get('page', 1)

        context = {
            'keyword': keyword,
            'category': category,
            'order': order,
            'page': page
        }

        return render(request, 'club/web/club-pr-posts-web.html', context)

    def post(self, request):
        datas = request.POST

        context = {
            'keyword': datas.get('keyword', ''),
            'category': datas.get('category', ''),
            'order': datas.get('order', '최신순'),
            'page': datas.get('page', 1)
        }

        return render(request, 'club/web/club-pr-posts-web.html', context)


class ClubPrPostListAPI(APIView):
    def post(self, request):
        data = request.data

        keyword = data.get('keyword', '')
        page = int(data.get('page', 1))
        category = data.get('category', '')
        ordering = data.get('order', '최신순')
        ordering = '-id' if ordering == '최신순' else '-view_count'

        row_count = 6
        offset = (page - 1) * row_count
        limit = page * row_count

        condition = Q()
        condition &= Q(post_title__icontains=keyword) | Q(club__club_name__icontains=keyword)
        condition &= Q(category__category_name__icontains=category)

        total_count = ClubPost.enabled_objects.filter(condition).count()

        page_count = 5
        end_page = math.ceil(page / page_count) * page_count
        start_page = end_page - page_count + 1
        real_end = math.ceil(total_count / row_count)
        end_page = real_end if end_page > real_end else end_page

        if end_page == 0:
            end_page = 1

        page_info = {
            'totalCount': total_count,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        club_posts = list(ClubPost.enabled_objects.filter(condition).values().order_by(ordering)[offset:limit])

        for club_post in club_posts:
            club_post['category_name'] = Category.objects.filter(id=club_post['category_id']).first().category_name
            club_post['club_name'] = Club.objects.filter(id=club_post['club_id']).first().club_name
            club_post['club_member_count'] = ClubMember.enabled_objects.filter(club_id=club_post['club_id']).count()
            club_post['reply_count'] = ClubPostReply.enabled_objects.filter(club_post_id=club_post['id']).count()

        club_posts.append(page_info)

        return Response(club_posts)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# 클럽 상세보기 페이지에서 틴플레이를 선택했을 때 화면에 보여지는 View
class ClubTeenplayAPIView(APIView):
    def get(self, request, club_id, page):
        # 최대 5개까지 썸네일을 보여주기 위한 count
        row_count = 5
        # DB에서 값을 가져올 떄 offset 부터 linit 까지 가져오기 위한 수식
        offset = (page - 1) * row_count
        limit = page * row_count

        # teenplay_list의 경우 페이지의 숫자만큼 늘어나면서 가져와야 하기 때문에 [offset:limit] 슬라이싱 사용
        # has_next의 경우 다음 영상 개수가 1개라도 존재하는지 아닌지 확인하기 위해서 exists() 함수 사용
        context = {
            'member': request.session['member'],
            'club': Club.objects.filter(id=club_id).values(),
            'teenplay_list': TeenPlay.enable_objects.filter(club=club_id).annotate(like_count=Count('teenplaylike__status')).values('like_count','id', 'created_date', 'updated_date','teenplay_title','club_id','video_path','thumbnail_path','status','club__member_id').order_by('-id')[offset:limit],
            'has_next': TeenPlay.enable_objects.filter(club=club_id)[limit:limit + 1].exists()
        }
        return Response(context)


# 틴플레이 삭제를 눌렀을 때 작동하는 REST API
class ClubTeenplayDeleteAPIView(APIView):
    # DB에서 삭제가 아닌 Soft Delete 를 사용하기 때문에 update 메소드 사용
    # 페이지 오류 또는 특이사항 발생 시 DB가 변경이 되지 않도록 transaction.atomic 데코레이터 사용
    @transaction.atomic
    def get(self, request, teenplay_id):
        TeenPlay.enable_objects.filter(id=teenplay_id).update(status=0)
        return Response("success")


# 틴플레이 영상 업로드 진행 시 사용되는 REST API
class ClubTeenplayUploadAPIView(APIView):
    # 페이지 오류 또는 특이사항 발생 시 DB가 변경이 되지 않도록 transaction.atomic 데코레이터 사용
    # 영상 정보에 대한 많은 정보가 담겨 있을 수 있기 때문에 get 방식이 아니라 post 방식으로 전달받음
    @transaction.atomic
    def post(self, request):
        # 영상 title 에 입력한 정보를 받기 위해서 post 방식으로 전달 받음
        data = request.POST
        # 영상 정보 (image, video) 파일은 request.FILES 객체에 담겨져 있기 때문에 해당 정보 변수 대입
        files = request.FILES

        # request 에 담긴 정보는 모두 딕셔너리 형태로 저장되어 있기 때문에 data 딕셔너리에 신규로 각각의 값 생성
        data = {
            'teenplay_title': data['title'],
            'club_id': data['clubId'],
            'video_path': files['video'],
            'thumbnail_path': files['thumbnail']
        }

        # 영상 생성이 될수 있도록 create 메소드 사용, data 딕셔너리에 있는 값은 unpacking 기능을 이용하여 create 변수 대입
        TeenPlay.objects.create(**data)
        return Response("success")
