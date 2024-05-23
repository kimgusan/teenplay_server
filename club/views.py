import math
import os.path

from django.db import transaction
from django.db.models import F, Q, Count
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views import View
from rest_framework.response import Response
from rest_framework.views import APIView

from activity.models import Activity, ActivityLike
from alarm.models import Alarm
from club.models import Club, ClubMember, ClubPost, ClubPostReply, ClubNotice, ClubCategory
from member.models import Member
from teenplay.models import TeenPlay
from teenplay_server import settings
from teenplay_server.category import Category
from teenplay_server.models import Region
from teenplay_server.utils.util.util import check_the_comments


# 모임 생성 전 모임에 대한 소개글을 정적 데이터로 구성해 놓은 페이지로 이동하는 view
class ClubIntroView(View):
    # 페이지 내 데이터가 모두 정적 데이터 임으로 이동만 실행 한다.
    def get(self, request):
        return render(request, 'club/web/club-intro-web.html')


# 모임 생성 페이지로 이동, 생성 해주는 메소드를 갖고 있는 view
class ClubCreateView(View):
    # 모임 생성 시 필요한 정보는 session과 생성 페이지에서 입력받기 때문에 다른 정보는 전달하지 않는다.
    def get(self, request):

        categories = Category.objects.all()
        regions = Region.objects.all()

        return render(request, 'club/web/club-create-web.html', {'categories': categories, 'regions': regions})

    # 모임 생성 후 상세보기 페이지로 이동하지만 request를 비워 반복 생성을 막기 위해 redirect를 사용
    @transaction.atomic
    def post(self, request):
        # form태그의 post메소드 속성으로 전달 받은 데이터를 저장
        data = request.POST
        # form태그의 post메소드 속성으로 전달 받은 file데이터를 저장
        file = request.FILES
        # club에 insert할 때 회원의 정보를 객체로 전달하기 위해 session에 있는 회원의 정보를 객체화
        member = Member(**request.session['member'])
        # 새로운 모임 데이터를 만들기 위해 필요한 정보들을 dict객체 형식으로 만들기
        data = {
            'club_main_category_id': data['club-category'],
            'club_name': data['club-name'],
            'club_region_id': data['club-region'],
            'club_intro': data['club-intro'],
            'member': member,
            'club_profile_path': file.get('club-profile'),
            'club_banner_path': file.get('club-banner')
        }

        # kwagrs형식으로 사용하여 club테이블에 insert하고 Club타입의 메소드를 사용하기 위해 객체화 진행
        club = Club.objects.create(**data)

        # 생성한 모임의 서브 카테고리를 저장하기 위해 생성한 모임의 id를 객체화
        club_id = club.id
        # 서브 카테고리를 저장하기 위해 입력 받은 정보를 저장
        sub_categories = request.POST.getlist('club-sub-category')

        if sub_categories is not None:
            for sub_category in sub_categories:
                # 새로운 모임 데이터를 만들기 위해 필요한 정보들을 dict객체 형식으로 만들기
                data = {
                    'club_id': club_id,
                    'category_id': sub_category
                }
                # kwagrs형식으로 사용하여 clubCategory테이블에 insert
                ClubCategory.objects.create(**data)

        # 새로고침 할 때마다 insert되는 것을 방지하기 위해 redirect방식을 사용하였으며
        # 모임 상세보기 이동은 간단한 url이기 때문에 Model Convention의 def get_absolute_url()을 정의하고 사용
        return redirect(club.get_absolute_url())


class ClubDetailView(View):
    def get(self, request):
        club_id = request.GET['id']
        # 상세보기로 이동 시 공지사항, 틴플레이 더보기를 클릭 해서 왔을 경우 바로 해당 정보를 보여주기 위한 구분점
        view = request.GET.get('view', 'activity')
        # 화면에서 필요한 컬럼들을 가독성을 높이기 위해 list형식으로 담아 사용
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

        # club_id를 통해 모임을 select 한 후 .annotate().values()를 사용해서 모임장의 정보를 각 쿼리셋 객체의 필드에 추가하고
        # 모임장의 정보는 정참조를 통해, 모임 구성원의 수는 역참조를 통해 club_member의 status가 1인 것만 카운트하여 필드에 추가
        club_list = Club.objects.filter(id=club_id) \
            .annotate(
                owner_id=F('member__id'),
                owner_name=F('member__member_nickname'),
                owner_email=F('member__member_email'),
                owner_phone=F('member__member_phone')
            ).values(*columns).annotate(
                    club_member_count=Count('clubmember', filter=Q(clubmember__status=1)))
        # 위 코드에 이어서 사용할 경우 중복이 발생하여 집계가 제대로 되지 않기 때문에 모임의 활동수는 따로 집계하도록 구성
        # .distinct()를 사용하지 않은 이유는 group_by가 된 경우 중복 제거가 제대로 되지 않기 때문
        club_activity_count = Club.objects.filter(id=club_id).values('id') \
            .annotate(club_activity_count=Count('activity')).first()
        # javscript에서 사용하기 편하게 하기 위해 쿼리셋 객체를 list로 형변환
        club_list = list(club_list)
        # 중복 제거를 위해 별도로 구한 구성원 수를 club 필드에 추가
        for club in club_list:
            club['club_activity_count'] = club_activity_count.get('club_activity_count')
            club['view'] = view
            # club_info가 비어 있을 때 화면에서 None으로 나오는 경우가 있어 빈문자열로 변경
            if club['club_info'] is None:
                club['club_info'] = ''

        return render(request, 'club/web/club-detail-web.html', {'club_list': club_list})


# 모임 상세보기의 관리하기, 가입신청, 신청취소, 탈퇴하기 버튼을 관리하는 API
class ClubMemberAPI(APIView):
    # 회원과 모임의 id를 통해 정보를 가져와서 전해주는 REST api
    def get(self, request, member_id, club_id):
        # 해당 모임에 가입한적이 없더라도 문제가 없도록 하기 위해 filter를 사용
        club_members = ClubMember.objects.filter(member=member_id, club=club_id).values()
        # fetch로 정보에 따라 화면에 보여주시위해 객체를 리턴
        return Response(club_members)

    # 관리하기, 가입신청, 신청취소, 탈퇴하기 버튼 클릭 시 club_member에서 추가 또는 수정하고
    # 상태에 따라 모임장에게 알림을 보내는 REST api
    @transaction.atomic
    def patch(self, request, member_id, club_id):
        member = Member.objects.get(id=member_id)
        club = Club.objects.get(id=club_id)
        # 모임에 가입한적이 없는 회원일 수 있기 때문에 get_or_create를 사용
        club_member, created = ClubMember.objects.get_or_create(member=member, club=club)
        # 기능이 제대로 작동하는지 확인하기 위해 선언한 변수
        message = 'create-apply'
        # 알람 전송 여부를 판단하기 위해 선언한 변수
        flag = True
        # 알람 전송 시 필요한 타입을 선언
        alarm_type = 9
        # 만약 get 이라면
        if not created:
            # 모임 구성원의 상태가 신청중인 상태라면
            if club_member.status == -1:
                # 탈퇴 상태로 변경
                club_member.status = 0
                # updated_date를 현재 시간으로 변경
                club_member.updated_date = timezone.now()
                # save()를 통해 update쿼리 실행
                club_member.save(update_fields=['status', 'updated_date'])

                message = 'cancel'
                # 가입신청 취소는 알림을 전송하지 않기 때문에 False로 변경
                flag = False
            # 모임 구성원의 상태가 탈퇴 상태라면
            elif club_member.status == 0:
                # 신청중 상태로 변경
                club_member.status = -1
                club_member.updated_date = timezone.now()
                club_member.save(update_fields=['status', 'updated_date'])
                # 가입신청을 보냈을 때 알림이 전송되야 하기 때문에 flag, alarm_type 변경없이 진행
                message = 'apply'
            # 모임 구성원의 상태가 가입 상태라면
            elif club_member.status == 1:
                # 탈퇴 상태로 변경
                club_member.status = 0
                club_member.updated_date = timezone.now()
                club_member.save(update_fields=['status', 'updated_date'])

                message = 'quit'
                # 탈퇴를 했을 때 알람은 전송하지만 alarm_type이 다르기 때문에 alarm_type만 변경 후 진행
                alarm_type = 10
        # 만약 flag가 True라면
        if flag:
            # 알람의 대상은 club, 알림 종류는 위에서 선언 되거나 변경된 alarm_type,
            # 알람을 보내는 사람은 로그인한 회원, 알람을 받는 사람은 모임장이기 때문에 club.member로 정보를 전달하여 insert 실행
            Alarm.objects.create(target_id=club_id, alarm_type=alarm_type, sender=member, receiver=club.member)
        # 알고리즘이 제대로 이루어 졌는지 확인하기 위해 message를 리턴
        return Response(message)


# 모임에서 개설한 활동중 진행중인 행사를 가져오는 REST api
# 종료된 행사와 구분하여 화면에 보여줘야 하기 때문에 각각 REST api를 구현
class ClubOngoingActivityAPI(APIView):
    def get(self, request, club_id):
        member = Member(**request.session['member'])
        club = Club.objects.get(id=club_id)
        # 해당 모임에서 개설한 활동 중 활동 종료시간이 현재시간보다 나중이고 취소하지 않은 활동들의
        # id, activity_title, thumbnail_path, activity_start에 대한 정보만 가져오고
        # activity_member에 역참조를 통해 각 활동에 참가 확정이 된 회원의 수를 세어 participant_count로 저장하고 list로 변환
        ongoing_activities = list(Activity.objects.filter(club=club, activity_end__gt=timezone.now(), status=1)
                                  .values('id', 'activity_title', 'thumbnail_path', 'activity_start')
                                  .annotate(participant_count=Count('activitymember', filter=Q(activitymember__status=1))))
        # 반복문을 통해 ongoing_activities의 각 ongoing_activity에 접근하여
        # is_like라는 필드명으로 로그인한 회원이 각 활동을 관심 목록에 추가 했는지 확인하고 값으로 저장
        for ongoing_activity in ongoing_activities:
            ongoing_activity['is_like'] = ActivityLike.enabled_objects.filter(activity=ongoing_activity['id'],
                                                                              member=member).exists()
        # 위 과정이 끝난 ongoing_activities를 화면에서 사용하기 위해 return
        return Response(ongoing_activities)

# 모임에서 개설한 활동중 종료된 행사를 가져오는 REST api
class ClubFinishedActivityAPI(APIView):
    # 현재까지 개설한 모든 활동중 종료된 행사들을 보여주기 때문에 pagination을 사용
    def get(self, request, club_id, page):
        member = Member(**request.session['member'])
        # 한 페이지당 보여줄 활동의 개수를 변수로 선언
        row_count = 8
        # 페이지 별로 시작점이 되는 인덱스를 변수로 선언
        offset = (page - 1) * row_count
        # 페이지 별로 끝나는 점이 되는 인덱스를 변수로 선언
        limit = page * row_count

        club = Club.objects.get(id=club_id)
        # 해당 모임에서 개설한 활동 중 활동 종료시간이 현재시간보다 이전이고 취소하지 않은 활동들의
        # id, activity_title, thumbnail_path, activity_start에 대한 정보만 가져오고 최신순으로 정렬
        # activity_member에 역참조를 통해 각 활동에 참가 확정이 된 회원의 수를 세어 participant_count로 저장하고 list로 변환
        finished_activities = list(Activity.objects.filter(club=club, activity_end__lte=timezone.now(), status=1)
                                   .values('id', 'activity_title', 'thumbnail_path', 'activity_start')
                                   .annotate(
            participant_count=Count('activitymember', filter=Q(activitymember__status=1)))
                                   .order_by('-id'))
        # 반복문을 통해 finished_activities 각 finished_activity 접근하여
        # is_like라는 필드명으로 로그인한 회원이 각 활동을 관심 목록에 추가 했는지 확인하고 값으로 저장
        for finished_activity in finished_activities:
            finished_activity['is_like'] = ActivityLike.enabled_objects.filter(activity=finished_activity['id'],
                                                                               member=member).exists()
        # 더보기 클릭 시 페이지네이션 처리된 정보가 넘어가야하기 때문에 처리하여 리턴
        return Response(finished_activities[offset:limit])


# 모임의 공지사항을 가져오는 REST api
class ClubNoticeAPI(APIView):
    # 전달 받은 정보를 통해 조회 후 결과를 페이지네이션 처리하여 리턴하는 메소드
    def get(self, request, club_id, page):
        # 한 페이지 당 보여줄 공지사항의 개수를 변수로 선언
        row_count = 4
        offset = (page - 1) * row_count
        limit = page * row_count

        # 해당 모임이 작성한 모임 공지사항이면서 삭제하지 않은 공지사항을 가져와 최신순으로 정렬
        club_notices = ClubNotice.objects.filter(club_id=club_id, status=1).values().order_by('-id')
        # 조회한 정보를 페이지네이션 처리하여 리턴
        return Response(club_notices[offset:limit])

# 모임 홍보글 작성에 대한 페이지 이동 및 inserte하는 view
class ClubPrPostWriteView(View):
    # 작성 시 모임과 모임명을 함께 전달하면서 페이지 이동
    def get(self, request):
        club = Club.objects.get(id=request.GET['club_id'])
        # 화면에서 사용할 모임명과 inserte할 때 필요한 모임 id를 담는다.
        context = {
            'club_id': club.id,
            'club_name': club.club_name,
        }
        # 정보 전달 및 이동만 하기 때문에 render방식 사용
        return render(request, 'club/web/club-pr-posts-write-web.html', context)

    @transaction.atomic
    def post(self, request):
        # form태그를 통해 전달 받은 정보를 각 방식에 맞게 전달받고 각각 변수로 저장
        datas = request.POST
        files = request.FILES
        club = Club.objects.get(id=request.POST['club_id'])
        # # 카테고리의 이름을 통해 조회하고 객체화
        # category = Category.objects.get(category_name=datas['category'])

        # club_post에 insert하기 위해 필요한 정보를 dict형태로 선언
        datas = {
            'club': club,
            'post_title': datas['title'],
            'post_content': datas['content'],
            # 'category': category,
            'image_path': files['image']
        }
        # 위에서 선언한 변수를 언페킹하여 create에 사용
        club_post = ClubPost.objects.create(**datas)

        return redirect(club_post.get_absolute_url())


# 모임 홍보글 상세보기로 이동하는 view
class ClubPrPostDetailView(View):
    # 현재 페이지 이동 시 댓글들도 함께 조회 후 전달하는 방식이지만 모임 홍보글에 대한 정보만 전달하고
    # REST api를 통해 스크롤이 특정 높이 일때 댓글 목록을 가져와 화면에 보이도록 하는 방식으로 변경하는 것이 더 효율적일 수 있다고 판단됨
    def get(self, request):
        # 해당 모임 홍보들에 대한 정보를 조회 후 객체화
        club_post = ClubPost.enabled_objects.get(id=request.GET['id'])
        # 해당 모임에 대한 카테고리를 조회 후 객체화
        club = Club.objects.get(id=club_post.club_id)
        club_main_category_id = club.club_main_category_id
        category = Category.objects.get(id=club_main_category_id)
        # 해당 모임 홍보글에 대한 댓글들을 조회 후 객체화, 없을 경우 오류를 방지하기 위해 .filter()사용
        replies = ClubPostReply.enabled_objects.filter(club_post=club_post).values()
        # 해당 모임 홍보글의 조회수를 1증가
        club_post.view_count += 1
        # updated_date를 현재시간으로 변경
        club_post.updated_date = timezone.now()
        club_post.save(update_fields=['view_count', 'updated_date'])

        context = {
            'club_post': club_post,
            'club_category': category,
            'replies': list(replies)
        }

        return render(request, 'club/web/club-pr-posts-detail-web.html', context)


# 모임 홍보글 수정 페이지 이동 및 수정하는 메소드를 갖고있는 view
class ClubPrPostUpdateView(View):
    # 수정 페이지에서 보여줘야 할 기존 작성한 정보들을 담아서 이동
    def get(self, request):
        # 쿼리스트링으로 전달 받은 홍보글 아이디를 변수로 선언
        club_post_id = request.GET['id']
        # 수정할 모임 홍보글에 대한 정보를 객체화
        club_post = ClubPost.enabled_objects.get(id=club_post_id)
        # 이미지가 수정 될 경우 배포서버 내 저장 되어 있는 기존 이미지를 삭제하기 위해 전체 경로를 만들어 변수로 선언
        file_path = os.path.join(settings.MEDIA_ROOT, club_post.image_path.path)
        # 화면에서 이미지 파일에 대한 size가 필요하여 위 경로를 기준으로 size를 구하고 변수로 선언
        file_size = os.path.getsize(file_path)

        context = {
            'club_post': club_post,
            'file_size': file_size
        }

        return render(request, 'club/web/club-pr-posts-update-web.html', context)

    # form태그를 통해 전달 받은 정보를 기준으로 모임 홍보글 수정
    def post(self, request):
        datas = request.POST
        files = request.FILES

        club_post = ClubPost.objects.get(id=datas['club_post_id'])
        # category = Category.objects.get(category_name=datas['category'])

        club_post.post_title = datas['title']
        club_post.post_content = datas['content']
        # club_post.category = category
        # 만약 전달 받은 file정보가 있다면
        if files:
            # ClubPost class에 선언해 놓은 image_delete() 메소드를 사용해 기존 이미지를 삭제
            club_post.image_delete()
            # 새롭게 전달받은 이미지 경로로 변경
            club_post.image_path = files['image']
            club_post.updated_date = timezone.now()
            club_post.save(update_fields=['post_title', 'post_content','image_path', 'updated_date'])
        # 만약 전달 받은 file정보가 없다면
        else:
            club_post.updated_date = timezone.now()
            club_post.save(update_fields=['post_title', 'post_content', 'updated_date'])

        return redirect(club_post.get_absolute_url())


# 여러 법적 문제를 대비한다는 전제하에 soft-delete로 기획
class ClubPrPostDeleteView(View):
    def post(self, request):
        datas = request.POST
        club_post = ClubPost.objects.get(id=datas['id'])
        # 기획 시 모든 테이블은 status라는 컬럼을 갖고 0이 삭제, 1은 게시중으로 통일
        club_post.status = 0
        club_post.updated_date = timezone.now()
        club_post.save(update_fields=['status', 'updated_date'])

        return redirect('/club/pr-post-list/')


# 모임 홍보글 댓글에 대한 REST api
class ClubPrPostReplyAPI(APIView):
    # 모임 홍보글 댓글의 목록 정보를 가져와 페이지네이션해주는 메소드
    def get(self, request):
        # 쿼리스트링으로 전달받은 page를 가져오고 만약 없다면 1을 넣어준다.
        page = int(request.GET.get('page', 1))
        club_post_id = request.GET.get('club_post_id')

        # 한 페이지 당 댓글의 개수를 변수로 선언
        row_count = 4
        offset = (page - 1) * row_count
        limit = page * row_count
        # soft-delete로 인해 .filter(status=1)을 주로 사용하기 때문에 manages에 따로 선언 후 모델에 enabled_objects라는 이름으로 선언
        # ClubPostReply에서 조회 후 작성자에 대한 정보는 .annotate()에 F표현식을 사용하여 member에 역참조하여 조회 후 각 필드를 추가하고 최신순으로 정렬
        replies = ClubPostReply.enabled_objects.filter(club_post_id=club_post_id) \
            .annotate(member_email=F('member__member_email'), member_name=F('member__member_nickname'),
                      member_path=F('member__memberprofile__profile_path')) \
            .values('id', 'reply_content', 'created_date', 'member_id', 'member_email', 'member_name',
                    'member_path').order_by('-id')
        # 댓글의 전체 개수를 따로 변수로 선언
        replies_count = replies.count()
        replies_info = {
            'replies': replies[offset:limit],
            'replies_count': replies_count
        }

        return Response(replies_info)

    # 새로운 댓글을 작성해주는 메소드
    @transaction.atomic
    def post(self, request):
        # JSON형식으로 전송한 데이터를 변수로 선언
        data = request.data

        data = {
            'reply_content': data['reply_content'],
            'club_post_id': data['club_post_id'],
            'member_id': request.session['member']['id']
        }

        result = check_the_comments(data['reply_content'])

        if result == 'profanity':
            return Response(result)

        post_reply = ClubPostReply.objects.create(**data)

        # 알림을 받을 회원의 id를 알기위해 club_post_id로 조회
        club_post = ClubPost.enabled_objects.get(id=data['club_post_id'])

        data = {
            'target_id': club_post.id,
            'alarm_type': 1,
            'sender_id': post_reply.member_id,
            'receiver_id': club_post.club.member_id
        }

        # 만약 알람을 보내는 이와 모임장이 동일하지 않다면
        if data['sender_id'] != data['receiver_id']:
            # 알람 객체 생성
            Alarm.objects.create(**data)

        return Response("success")

    # 댓글 작성자가 댓글 수정을 실행하면 응답하는 메소드
    @transaction.atomic
    def patch(self, request):
        data = request.data
        reply_content = data['reply_content']
        reply_id = data['id']

        result = check_the_comments(reply_content)

        if result == 'profanity':
            return Response(result)

        # 전달 받은 댓글 id를 통해 수정할 댓글 조회
        club_post_reply = ClubPostReply.enabled_objects.get(id=reply_id)
        club_post_reply.reply_content = reply_content
        club_post_reply.updated_date = timezone.now()
        club_post_reply.save(update_fields=['reply_content', 'updated_date'])

        return Response("success")

    # 댓글 작성자가 댓글 삭제를 실행하면 응답하는 메소드
    @transaction.atomic
    def delete(self, request):
        reply_id = int(request.GET.get('id'))

        # 전달 받은 댓글 id를 통해 수정할 댓글 조회
        club_post_reply = ClubPostReply.enabled_objects.get(id=reply_id)
        # soft-delete를 사용하기 때문에 status 변경
        club_post_reply.status = 0
        club_post_reply.updated_date = timezone.now()
        club_post_reply.save(update_fields=['status', 'updated_date'])

        return Response("success")


# 모임 홍보글 목록으로 이동하는 view
class ClubPrPostListView(View):
    # 검색 기록없이 이동 시 사용하는 메소드
    def get(self, request):
        # 검색과 관련된 사항들을 변수로 선언, 비어 있을 경우 기본값 또는 빈 문자열이 들어가도록 설정
        keyword = request.GET.get('keyword', '')
        category = request.GET.get('category', '')
        order = request.GET.get('order', '최신순')
        page = request.GET.get('page', 1)

        categories = Category.objects.all()

        context = {
            'keyword': keyword,
            'category': category,
            'order': order,
            'page': page,
            'categories': categories
        }

        return render(request, 'club/web/club-pr-posts-web.html', context)

    # 모임 홍보글에서 목록으로 버튼을 클릭 시 사용되는 메소드
    # get방식으로 이동 할 경우 url에 쿼리스트링으로 검색 관련 사항들이 표시되는것을 방지하기 위해 post방식으로 이동
    def post(self, request):
        datas = request.POST

        categories = Category.objects.all()

        context = {
            'keyword': datas.get('keyword', ''),
            'category': datas.get('category', ''),
            'order': datas.get('order', '최신순'),
            'page': datas.get('page', 1),
            'categories': categories,
        }

        return render(request, 'club/web/club-pr-posts-web.html', context)


# 카테고리, 키워드, 페이지, 정렬에 대한 정보를 받고 정보를 보내주는 REST api
class ClubPrPostListAPI(APIView):
    def get(self, request):
        data = request.GET

        keyword = data.get('keyword', '')
        page = int(data.get('page', 1))
        category = data.get('category', '')
        ordering = data.get('order', '최신순')
        ordering = '-id' if ordering == '최신순' else '-view_count'

        # 한 페이지 당 6개의 홍보글을 보여준다.
        row_count = 6
        offset = (page - 1) * row_count
        limit = page * row_count
        # 조건식에 icontains을 넣음으로써 대,소문자 상관없이 조회하도록 하였으며 keyword는 홍보글 제목이나 모임 이름이 둘중 하나라도 포함되어도 문제없도록 구성
        condition = Q(post_title__icontains=keyword) | Q(club__club_name__icontains=keyword)

        # category의 이름으로 조회
        if category:
            condition &= Q(club__club_main_category_id__exact=category)
        # 위에서 작성한 조건식으로 검색된 정보들의 개수를 집계하고 변수로 선언
        total_count = ClubPost.enabled_objects.filter(condition).count()

        # 숫자 버튼 형식의 페이지네이션 처리를 위해 작성된 코드
        page_count = 5
        end_page = math.ceil(page / page_count) * page_count
        start_page = end_page - page_count + 1
        real_end = math.ceil(total_count / row_count)
        # 실제 끝 페이지와 계산된 페이지가 다른 경우를 위한 삼항 연산자
        end_page = real_end if end_page > real_end else end_page
        # 만약 마지막 페이지가 0이라면 1이 되도록 변경
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
        # 조건식, 정렬 방식, 페이지네이션 처리를 한 쿼리셋 객체를 list형식으로 바꾸어 변수 선언
        club_posts = list(ClubPost.enabled_objects.filter(condition).values().order_by(ordering)[offset:limit])
        # 반복문을 이용하여 각 모임 홍보글에 화면에서 필요로하는 정보를 추가
        for club_post in club_posts:
            # 여러 테이블에서 정보를 가져와야 하기 때문에 역참조를 사용할 경우 중복이 발생하기 때문에 각각 필요한 정보만 가져와 값을 넣는 방식을 선택
            # club_post['category_name'] = Category.objects.filter(id=club_post['category_id']).first().category_name
            club_post['club_name'] = Club.objects.filter(id=club_post['club_id']).first().club_name
            club_post['club_member_count'] = ClubMember.enabled_objects.filter(club_id=club_post['club_id']).count()
            club_post['reply_count'] = ClubPostReply.enabled_objects.filter(club_post_id=club_post['id']).count()

            club_id = club_post['club_id']
            club_category_id = Club.objects.filter(id=club_id).first().club_main_category_id
            club_post['category_name'] = Category.objects.filter(id=club_category_id).first().category_name

        club_posts.append(page_info)

        return Response(club_posts)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class ClubTeenplayAPIView(APIView):
    def get(self, request, club_id, page):
        row_count = 5
        offset = (page - 1) * row_count
        limit = page * row_count

        context = {
            'member': request.session['member'],
            'club': Club.objects.filter(id=club_id).values(),
            'teenplay_list': TeenPlay.enable_objects.filter(club=club_id).annotate(like_count=Count('teenplaylike__status')).values('like_count','id', 'created_date', 'updated_date','teenplay_title','club_id','video_path','thumbnail_path','status','club__member_id').order_by('-id')[offset:limit],
            'has_next': TeenPlay.enable_objects.filter(club=club_id)[limit:limit + 1].exists()
        }
        return Response(context)


class ClubTeenplayDeleteAPIView(APIView):
    @transaction.atomic
    def get(self, request, teenplay_id):
        TeenPlay.enable_objects.filter(id=teenplay_id).update(status=0)
        return Response("success")


class ClubTeenplayUploadAPIView(APIView):
    @transaction.atomic
    def post(self, request):
        data = request.POST
        files = request.FILES

        data = {
            'teenplay_title': data['title'],
            'club_id': data['clubId'],
            'video_path': files['video'],
            'thumbnail_path': files['thumbnail']
        }

        TeenPlay.objects.create(**data)
        return Response("success")