import math
import os
from pathlib import Path

import joblib
from django.db import transaction
from django.db.models import F, Q
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views import View
from rest_framework.response import Response
from rest_framework.views import APIView

from activity.models import Activity, ActivityImage, ActivityMember, ActivityReply, ActivityLike
from alarm.models import Alarm
from club.models import Club, ClubMember, ClubNotice
from festival.models import Festival
from member.models import Member, MemberProfile
from pay.models import Pay
from search.models import RecentSearch
from teenplay_server import settings
from teenplay_server.category import Category
import re

from teenplay_server.models import Region
from bootpay_backend import BootpayBackend
import pickle

def make_datetime(date, time="00:00"):
    '''
    문자열 타입의 날짜 혹은 날짜와 시간을 전달 받아,
    datetime 객체를 반환하는 함수입니다.
    Datepicker api를 활용하여 입력받은 날짜를 문자열로 view에서 전달받은 후
    datetime 타입의 mysql 테이블 컬럼에 넣기 위해 사용합니다.
    전달받는 날짜의 형식은 '01/01/2000' 의 형식이며,
    이를 '연-월-일 시-분-초' 의 형식으로 변환합니다.
    '''
    date = date.split("/")
    date = "-".join([date[2], date[0], date[1]])
    time = time + ":00"
    datetime = timezone.datetime.strptime(date + " " + time, '%Y-%m-%d %H:%M:%S')
    return datetime


class ActivityCreateWebView(View):
    # 활동 개설 페이지로 이동하는 메소드입니다.
    def get(self, request):
        # 각 모임마다 해당 모임의 모임장만 활동을 개설할 수 있으므로,
        # 쿼리스트링으로 club_id를 받습니다.
        club_id = request.GET['club_id']
        # 지역축제 상세 페이지에서 해당 지역축제와 관련된 활동을 개설할 경우
        # 쿼리스트링으로 festival_id를 받습니다.
        festival_id = request.GET.get('festival_id')

        # tbl_club 테이블에서 해당 id에 맞는 모임 객체를 불러옵니다.
        club = Club.enabled_objects.filter(id=club_id).first()
        # 활동 개설 페이지에서 필요한 카테고리 객체들을 tbl_category 테이블에서 불러옵니다.
        categories = Category.objects.filter(status=True)

        # 화면에서 django template을 통해 사용할 수 있도록 context에 담아줍니다.
        context = {
            'club': club,
            'categories': categories,
            'festival_id': festival_id,
        }
        return render(request, 'activity/web/activity-new-web.html', context=context)

    # 활동 개설 버튼 클릭 후 결제 완료 시 호출되는 post 메소드입니다.
    @transaction.atomic
    def post(self, request):
        data = request.POST
        # post 형식으로 받아온 데이터에서 필요한 정보를 추출합니다.
        # 활동을 개설하는 모임의 id를 통해 모임 객체를 불러옵니다.
        club = Club.enabled_objects.get(id=data.get('club-id'))
        # 활동 개설 시 정보 입력 -> 결제완료 -> 결제정보 insert(pay/views.py)의 흐름이 먼저 진행되며,
        # 이때 tbl_pay에 insert된 결제 정보의 id가 post 형식의 데이터에 함께 담겨서 들어옵니다.
        pay_id = data.get('pay-id')
        pay = Pay.objects.filter(status=True, id=pay_id).first()
        # 아래부터는 활동 개설 시에 필요한 활동 관련 정보들을 data로부터 추출하여 적절히 다듬는 작업입니다.
        # 이때 오류 발생 시 바로 결제 취소 요청을 진행할 수 있도록 try/except 블록으로 묶습니다.
        try:
            # summernote에서 작성한 활동 내용을 불러옵니다.
            activity_content = data.get('activity-content')

            # 모집 시작/종료 날짜 및 시간을 받아와 클래스 바깥에 선언해 둔 make_datetime()함수를 통해 datetime 객체로 변환합니다.
            recruit_start = make_datetime(data.get('recruit-start-date'), data.get('recruit-start-time'))
            recruit_end = make_datetime(data.get('recruit-end-date'), data.get('recruit-end-time'))

            # 활동 개설 시 사용자가 선택한 카테고리의 id를 받아와 tbl_category에서 해당 카테고리 객체를 불러옵니다.
            category = Category.objects.get(id=data.get('category'))

            # 활동 개설 시 사용자가 업로드한 활동 썸네일 및 배너 이미지를 request.FILES.get()을 통해 불러옵니다.
            thumbnail = request.FILES.get('thumbnail-path')
            banner = request.FILES.get('banner-path')

            # 모집과 동일하게 활동 시작/종료 시간을 받아와 datetime 객체로 변환합니다.
            activity_start = make_datetime(data.get('activity-start-date'), data.get('activity-start-time'))
            activity_end = make_datetime(data.get('activity-end-date'), data.get('activity-end-time'))

            # 사용자가 지역축제와 관련된 활동을 개설했을 경우를 고려하여 festival-id를 받아옵니다.
            festival_id = data.get('festival-id')

            # festival 객체를 담을 변수를 None으로 초기화합니다.
            festival = None

            # 지역축제와 관련된 활동을 개설할 경우 tbl_festival에서 해당 축제 객체를 불러와 담아줍니다.
            if festival_id is not None:
                festival = Festival.objects.get(id=festival_id)

            # 정리한 데이터들을 딕셔너리에 담아줍니다.
            data = {
                'club': club,
                'activity_title': data.get('activity-title'),
                'activity_content': activity_content,
                'recruit_start': recruit_start,
                'recruit_end': recruit_end,
                'category': category,
                'activity_intro': data.get('activity-intro'),
                'activity_address_location': data.get('activity-address-location'),
                'activity_address_detail': data.get('activity-address-detail'),
                'thumbnail_path': thumbnail,
                'banner_path': banner,
                'activity_start': activity_start,
                'activity_end': activity_end,
                'festival': festival,
                'pay': pay,
            }

            # 딕셔너리를 **kwargs에 전달하여 django orm을 통해 tbl_activity에 insert합니다.
            # 이때 반환된 객체를 activity에 담아줍니다.
            activity = Activity.objects.create(**data)

            # summernote editor에서 업로드한 이미지들은 다른 APIView를 통해 REST 방식으로 DB 및 서버에 저장됩니다.
            # 이 부분은 해당 APIView에 보다 자세히 설명하겠습니다.
            # 아래 코드는 업로드한 이미지들의 tbl_activity_image 테이블 상의 status를 1로 바꿔주며, fk로 해당 활동과 연결해줍니다.
            image_ids = request.POST.get('image-id')
            if image_ids:
                for image_id in image_ids:
                    activity_image = ActivityImage.objects.filter(id=image_id)
                    if activity_image.exists():
                        activity_image = activity_image.first()
                        activity_image.status = 1
                        activity_image.activity_id = activity.id
                        activity_image.updated_date = timezone.now()
                        activity_image.save(update_fields=['status', 'activity_id', 'updated_date'])

            # 모임 구성원 중 알림을 켜놓은 사람들을 대상으로 활동 개설 알림을 전송합니다.
            # tbl_alarm 테이블에 정보를 insert하며 이때 target_id에는 알림목록에서 클릭 시 활동 상세페이지로 이동할 수 있도록
            # 활동의 id를 담아줍니다.
            alarmed_member_ids = ClubMember.objects.filter(status=True, club_id=club.id, alarm_status=1).values('member_id')
            for alarmed_member_id in alarmed_member_ids:
                id = alarmed_member_id.get('member_id')
                member = Member.enabled_objects.filter(id=id)
                if not member.exists():
                    continue
                member = member.first()
                data = {
                    'target_id': activity.id,
                    'alarm_type': 6, # alarm_type는 alarm 앱의 models.py에 정의되어 있습니다.
                    'sender': Member(**request.session.get('member')),
                    'receiver': member,
                }
                Alarm.objects.create(**data)

            # redirect 경로는 activity model에 정의해둔 get_absolute_url() 메소드의 반환값인 활동 상세페이지 경로입니다.
            return redirect(activity.get_absolute_url())

        except:
            # 오류가 발생하면 나머지 부분은 transaction.atomic 데코레이터를 통해 rollback 하지만,
            # 결제 부분은 bootpay api를 사용하므로 직접 서버에 결제취소 요청을 보내주어야 합니다.
            receipt_id = pay.receipt_id
            bootpay = BootpayBackend('65e44626e57a7e001be37370',
                                     'NQmDRBsgOfziMiNXUEKrJGQ+YhXZncneSVG/auKihFA=')

            token = bootpay.get_access_token()

            if 'error_code' not in token:
                response = bootpay.cancel_payment(receipt_id=receipt_id,
                                                  cancel_price=pay.price,
                                                  cancel_username='관리자', cancel_message='취소됨')

            # 취소가 완료되면 pay객체의 status와 updated_date를 변경한 후 save()를 통해 테이블에 update합니다.
            pay.status = 0
            pay.updated_date = timezone.now()
            pay.save(update_fields=['status', 'updated_date'])

            # 이때에는 다시 활동 개설 페이지의 처음으로 돌아가도록 redirect합니다.
            return redirect(f'/activity/create?club_id={club.id}')


class ActivityDetailWebView(View):
    # 활동 상세페이지로 이동하는 메소드입니다.
    def get(self, request):
        # 쿼리스트링으로 활동의 id를 받아옵니다.
        activity_id = request.GET['id']
        # 받아온 id로 활동 객체를 조회하여 불러옵니다.
        activity = Activity.objects.filter(id=activity_id).first()

        # 활동 상세페이지 화면에서 활용하기 위해 해당 활동의 카테고리 객체와 모임 객체를 변수에 담아줍니다.
        category = activity.category
        club = activity.club

        # 활동 상세페이지는 기획 상 로그인한 유저만 조회할 수 있으므로 세션에서 사용자의 id를 받아와 저장합니다.
        member_id = request.session['member']['id']
        member = Member.enabled_objects.get(id=member_id)

        # 회원에 맞는 활동 추천 ai 모델을 불러옵니다.
        model_file_name = member.member_recommended_activity_model

        # path
        model_file_path = os.path.join(Path(__file__).resolve().parent.parent, model_file_name)

        # pkl 파일을 열어 객체 로드
        model = joblib.load(model_file_path)

        # 불러온 ai 모델에 추가 fit을 진행합니다.
        additional_X_train = [activity.category.category_name + activity.activity_title + activity.activity_intro + activity.activity_address_location]
        additional_y_train = [activity.category.id]

        additional_X_train_transformed = model.named_steps['count_v'].transform(additional_X_train)
        model.named_steps['mnnb'].partial_fit(additional_X_train_transformed, additional_y_train, classes=[i for i in range(1, 14)])

        # fit이 완료된 모델을 다시 같은 경로에 같은 이름으로 내보내줍니다.
        joblib.dump(model, member.member_recommended_activity_model)

        # 해당 활동의 구성원 수를 Queryset 객체의 count() 메소드를 통해 조회합니다.
        activity_member_count = ActivityMember.enabled_objects.filter(activity_id=activity_id).count()

        # 해당 활동이 모집 중인지 여부를 현재 시간과 비교하여 판단합니다.
        activity_recruit_check = activity.recruit_end >= timezone.now() >= activity.recruit_start

        # 해당 활동에 달린 댓글 객체들을 가져와 list 타입으로 형변환한 후 변수에 담아줍니다.
        activity_replies = list(ActivityReply.enabled_objects.filter(activity_id=activity_id))

        # 화면에서 "신청하기" 버튼과 "관리하기"(모임장일 경우) 버튼 중 어느 버튼을 표시할지를 체크할 플래그를 만들어줍니다.
        activity_member_check = activity.club.member_id == member_id

        # 댓글 객체마다 해당 댓글 작성자의 프로필 사진을 tbl_member_profile에서 조회하여 member_profile이라는 필드명으로 담아줍니다.
        for reply in activity_replies:
            member_profile = MemberProfile.enabled_objects.filter(member_id=reply.member_id).first()
            reply.member_profile = member_profile

        # 해당 활동을 개설한 모임의 공지사항을 최신순으로 4개까지 조회하여 list로 형변환한 후 담아줍니다.
        club_notices = list(ClubNotice.objects.filter(status=True, club_id=club.id).order_by('-id')[:4])

        # 추천 활동 목록에 표시할 활동들을 가져옵니다. 이때 현재 보고 있는 활동은 제외합니다.
        recommended_activities = list(Activity.enabled_objects.exclude(id=activity_id)[0:4])

        for recommended_activity in recommended_activities:
            # 각 추천활동에 표시할 활동 참가자 수를 가져와 member_count 필드를 만들어 담아줍니다.
            member_count = ActivityMember.enabled_objects.filter(activity_id=recommended_activity.id).count()
            recommended_activity.member_count = member_count
            # 현재 로그인한 사용자가 각 추천활동마다 관심활동에 추가했는지 여부를 가져와 is_liked 필드를 만들어 담아줍니다.
            is_liked = ActivityLike.enabled_objects.filter(activity_id=recommended_activity, member_id=member_id).exists()
            recommended_activity.is_liked = is_liked

        # 현재 로그인한 사용자가 현재 보고 있는 활동을 관심활동에 추가했는지 여부를 가져와 저장합니다.
        is_like = ActivityLike.enabled_objects.filter(activity_id=activity_id, member_id=member_id).exists()

        # 위에서 정리한 데이터들을 화면에서 활용하기 위해 context에 담아줍니다.
        context = {
            'activity': activity,
            'category': category,
            'club': club,
            'activity_member_count': activity_member_count,
            'activity_recruit_check': activity_recruit_check,
            'activity_replies': activity_replies,
            'club_notices': club_notices,
            'recommended_activities': recommended_activities,
            'is_like': is_like,
            'activity_member_check': activity_member_check
        }

        return render(request, 'activity/web/activity-detail-web.html', context)


class ActivityLikeAPI(APIView):
    # 활동을 관심활동에 추가/삭제할 때 호출되는 REST API입니다.
    def get(self, request):
        # Javascript에서 fetch를 사용할 때 쿼리스트링으로 활동의 id를 받아와 저장합니다.
        activity_id = request.GET['id']

        # 관심활동에 추가하고자 하는 요청일 경우 is_create가 "true"로 전달됩니다.
        is_create = request.GET['is-create']

        # 현재 로그인한 사용자의 id를 세션에서 받아와 저장합니다.
        member_id = request.session['member']['id']

        # 관심활동에 추가하고자 하는 요청일 경우입니다.
        if is_create == "true":
            # get_or_create()를 통해 해당 사용자의 id와 해당 활동의 id로 만들거나 조회합니다.
            activity_like, created = ActivityLike.objects.get_or_create(activity_id=activity_id, member_id=member_id)
            # created가 False라면, 즉 이미 존재하지만 soft delete로 인해 status가 0이라면
            if not created:
                # status를 다시 1로 변경한 후 save()를 통해 update해줍니다.
                activity_like.status = 1
                activity_like.updated_date = timezone.now()
                activity_like.save(update_fields=['status', 'updated_date'])

            member = Member.enabled_objects.get(id=member_id)
            activity = Activity.enabled_objects.get(id=activity_id)

            # 회원에 맞는 활동 추천 ai 모델을 불러옵니다.
            model_file_name = member.member_recommended_activity_model

            # path
            model_file_path = os.path.join(Path(__file__).resolve().parent.parent, model_file_name)

            # pkl 파일을 열어 객체 로드
            model = joblib.load(model_file_path)

            # 불러온 ai 모델에 추가 fit을 진행합니다.
            additional_X_train = [
                activity.category.category_name + activity.activity_title + activity.activity_intro + activity.activity_address_location]
            additional_y_train = [activity.category.id]

            additional_X_train_transformed = model.named_steps['count_v'].transform(additional_X_train)
            model.named_steps['mnnb'].partial_fit(additional_X_train_transformed, additional_y_train,
                                                  classes=[i for i in range(1, 14)])

            # fit이 완료된 모델을 다시 같은 경로에 같은 이름으로 내보내줍니다.
            joblib.dump(model, member.member_recommended_activity_model)

            # 아래 코드로 내려가지 않도록 return해줍니다.
            return Response("added")

        # 위 if문에 작성한 조건식이 False일 경우, 즉 관심활동에서 삭제하고자 하는 요청일 경우
        # tbl_activity_like에서 해당 객체를 조회하여 가져옵니다.
        activity_like = ActivityLike.objects.filter(activity_id=activity_id, member_id=member_id).first()

        # 해당 객체의 status필드를 0으로 변경한 후 save()를 통해 update해줍니다.
        activity_like.status = 0
        activity_like.updated_date = timezone.now()
        activity_like.save(update_fields=['status', 'updated_date'])

        return Response("deleted")


class ActivityLikeCountAPI(APIView):
    # 활동을 관심활동에 등록한 사용자 수를 조회하는 REST API입니다.
    def get(self, request):
        # Javascript에서 fetch를 사용할 때 쿼리스트링으로 활동의 id를 받아와 저장합니다.
        activity_id = request.GET['id']

        # 해당 활동을 관심활동에 등록한 사용자 수를 Queryset 객체의 count() 메소드를 통해 집계하여 담아줍니다.
        activity_like_count = ActivityLike.enabled_objects.filter(activity_id=activity_id).count()

        # 이를 Response에 담아 return해줍니다.
        return Response(activity_like_count)

class ActivityMemberCountAPI(APIView):
    # 활동의 참가자 수를 조회하는 REST API입니다.
    def get(self, request):
        # Javascript에서 fetch를 사용할 때 쿼리스트링으로 활동의 id를 받아와 저장합니다.
        activity_id = request.GET['id']

        # 해당 활동의 참가자 수를 Queryset 객체의 count() 메소드를 통해 집계하여 담아줍니다.
        activity_member_count = ActivityMember.enabled_objects.filter(activity_id=activity_id).count()

        # 이를 Response에 담아 return해줍니다.
        return Response(activity_member_count)


# 댓글은 한 화면 내의 부분에서 페이지 새로고침 없이 작성, 조회, 수정, 삭제(CRUD)가 이루어지므로
# fetch를 통해 요청하고 REST를 통해 데이터로 응답합니다.
class ActivityReplyAPI(APIView):
    # 활동에 달린 댓글들을 페이지네이션을 적용하여 조회하는 REST API입니다.
    def get(self, request):
        # fetch 요청 경로에서 쿼리스트링으로 작성한 페이지를 받아와 정수로 형변환하며, 기본값을 1로 지정합니다.
        page = int(request.GET.get('page', 1))

        # 마찬가지로 쿼리스트링을 통해 해당 활동의 id를 받아옵니다.
        activity_id = request.GET.get('activity-id')

        # 한번에 표시할 댓글의 개수(행의 개수)를 3으로 지정합니다.
        row_count = 3

        # 현재 페이지에 표시할 첫 댓글의 인덱스입니다.
        offset = (page - 1) * row_count

        # 현재 페이지에 표시할 마지막 댓글의 인덱스입니다.
        limit = page * row_count

        # 댓글의 경우 숫자로 된 페이지 버튼이 아닌, 더보기 버튼으로 페이지네이션이 이루어집니다.

        # 해당 활동의 id로 댓글 목록을 최신순으로 조회합니다.
        # 댓글 작성자의 닉네임과 이메일을 tbl_member로 정방향 참조하여 가져옵니다.
        # 댓글 작성자의 프로필 사진 경로를 tbl_member로 정방향 참조 후 다시 tbl_member_profile로 역방향 참조하여 가져옵니다.
        # 그리고 화면에서의 편리한 사용을 위해 해당 컬럼의 이름을 annotate()를 통해 변경해준 후,
        # values()를 통해 필요한 정보들을 추출합니다.
        replies = ActivityReply.enabled_objects.filter(activity_id=activity_id).order_by('-id') \
            .annotate(member_nickname=F('member__member_nickname'),
                      member_path=F('member__memberprofile__profile_path'),
                      member_email=F('member__member_email'))\
            .values('reply_content', 'id', 'member_nickname', 'created_date', 'member_id', 'member_email')

        # 위에서 계산한 페이지네이션을 적용한 데이터를 Response에 담아 응답합니다.
        return Response(replies[offset:limit])

    # 댓글 작성 시 호출되는 REST API입니다.
    @transaction.atomic
    def post(self, request):
        # fetch 요청의 body에서 JSON.stringify()를 통해 전달 받은 댓글 데이터를 정리하여 딕셔너리에 담아줍니다.
        data = request.data
        data = {
            'reply_content': data['reply_content'],
            'activity_id': data['activity_id'],
            'member_id': data['member_id']
        }

        # 정리한 데이터를 통해 활동 댓글 객체를 생성하고, 반환된 객체를 저장합니다.
        # 이 객체는 응답하기 위한 객체가 아닌, 알람 생성 시 편리하게 사용하기 위한 객체입니다.
        activity_reply = ActivityReply.objects.create(**data)

        # 활동 id로 활동 객체를 가져와 저장합니다.
        activity = Activity.enabled_objects.filter(id=data['activity_id']).first()

        # 활동 객체의 필드인 모임 객체의 id로 한번 더 조회하여 모임 객체를 가져옵니다.
        club = Club.enabled_objects.filter(id=activity.club.id).first()

        # 해당 활동과 해당 모임이 None이라면, 즉 status가 0인 등의 이유로 조회가 되지 않았다면 아래 조건식이 False가 됩니다.
        # 그렇지 않고 둘 다 유효하다면 아래 if문 블록에 진입합니다.
        if activity and club:
            # 활동을 개설한 모임장에게 알람을 전송하기 위해 tbl_alarm에 insert할 정보를 담은 딕셔너리를 생성합니다.
            alarm_data = {
                'target_id': activity.id,
                'alarm_type': 2,
                'sender_id': activity_reply.member.id,
                'receiver_id': club.member.id,
            }
            # 이 딕셔너리를 통해 tbl_alarm에 insert합니다.
            Alarm.objects.create(**alarm_data)

        return Response("success")

    # 댓글 수정 시 호출되는 REST API입니다.
    @transaction.atomic
    def patch(self, request):
        # fetch 요청의 body에서 JSON.stringify()를 통해 전달 받은 댓글 데이터를 정리합니다.
        activity_id = request.data['activity_id']
        member_id = request.data['member_id']
        reply_content = request.data['reply_content']
        id = request.data['id']

        # 댓글 내용을 제외한 다른 정보들과 일치하는 댓글 객체를 조회합니다.
        activity_reply = ActivityReply.enabled_objects.get(id=id, activity_id=activity_id, member_id=member_id)

        # 해당 댓글 객체의 내용을 새로 받아온(수정된) 내용으로 변경한 후 save() 메소드를 통해 update해줍니다.
        activity_reply.reply_content = reply_content
        activity_reply.updated_date = timezone.now()
        activity_reply.save(update_fields=['reply_content', 'updated_date'])

        return Response("success")

    # 댓글 삭제 시 호출되는 REST API입니다.
    # 모든 삭제 서비스는 soft delete로 통일하여 실제 데이터베이스에서 delete하는 대신,
    # status를 update하는 방식으로 기획 및 구현했습니다.
    @transaction.atomic
    def delete(self, request):
        # fetch 요청의 body에서 JSON.stringify()를 통해 전달 받은 댓글 id입니다.
        id = request.data['id']

        # 해당 id에 맞는 댓글 객체를 조회합니다.
        activity_reply = ActivityReply.enabled_objects.get(id=id)

        # 해당 댓글 객체의 status를 0으로 바꿔준 후 save()를 통해 update합니다.
        activity_reply.status = 0
        activity_reply.updated_date = timezone.now()
        activity_reply.save(update_fields=['status', 'updated_date'])

        return Response("success")


# 활동 목록 페이지로 이동하는 view입니다.
# 비슷한 로직으로 get()과 post() 메소드 2개를 만든 이유는,
# 서비스 기획 상 활동 목록으로 이동할 수 있는 경우의 수가 여러가지이기 때문입니다.
# 먼저 get방식으로 요청하는 경우는 header의 내비게이션 바에서 활동목록을 클릭했을 경우나 헤더의 카테고리에서 전체를 클릭했을 경우 등
# parameter를 넘기지 않은 채 활동 목록 페이지로 이동할 경우입니다.
# post방식으로 요청하는 경우는 parameter와 함께 활동 목록으로 이동하고자 할 경우입니다.(검색, 각 카테고리 클릭 등)
# 쿼리스트링을 통해 get방식으로 요청하지 않는 이유는 아래와 같습니다.
# 1. 활동목록 페이지의 데이터들은 전부 fetch를 통해 비동기로 요청하여 화면에 출력합니다.
# 2. 활동목록 페이지에는 여러 가지 조회 필터가 존재하는데, 언제든 x버튼을 클릭하여 적용을 취소할 수 있습니다.
# 3. get방식의 쿼리스트링으로 parameter를 전달할 경우, 해당 필터를 적용 취소했을 때 uri에 쿼리스트링이 남아있지만
# 화면에 표시되는 조회 결과는 uri에 남아있는 parameter와 달라지게 됩니다.
# 4. 따라서 어떠한 경우에도 uri에 변화가 없도록 활동 목록 페이지로 이동하는 경우를 두 가지로 나누어 get방식과 post방식 모두 정의하였습니다.
class ActivityListWebView(View):
    # get 방식으로 parameter없이 요청할 경우입니다.
    def get(self, request):
        # post 방식의 경우 화면 쪽에서 javascript에서 필요한 값들이 있어,
        # 전달받은 값이 없더라도 빈 문자열로 초기화하여 담아줍니다.
        selected_category = ''
        keyword = ''

        # 화면에서 지역 필터 selectbox에 표시할 지역 리스트를 조회하여 담아줍니다.
        regions = Region.objects.filter(status=True)

        # 마찬가지로 카테고리 리스트를 조회하여 담아줍니다.
        categories = Category.objects.filter(status=True)

        # 해당 정보들을 context 딕셔너리에 넣어줍니다.
        context = {
            'selected_category': selected_category,
            'categories': categories,
            'regions': regions,
            'keyword': keyword
        }

        return render(request, 'activity/web/activity-web.html', context=context)

    # post 방식으로 parameter들과 함께 요청할 경우입니다.
    def post(self, request):
        # request.POST.get()을 사용하여 post로 전달받은 데이터에 없더라도 KeyError 없이 None이 담기게 됩니다.
        # selected_category는 카테고리를 클릭하여 활동 목록으로 넘어왔을 경우 해당 카테고리의 id가 담깁니다.
        selected_category = request.POST.get('category-id')

        # keyword는 검색어를 입력하였거나 추천 검색어 중 하나를 클릭하여 활동 목록으로 넘어왔을 경우 해당 단어가 담깁니다.
        keyword = request.POST.get('keyword', '')

        # get방식과 마찬가지로 지역 목록과 카테고리 목록을 담아줍니다.
        regions = Region.objects.filter(status=True)
        categories = Category.objects.filter(status=True)

        # 검색어가 있을 경우 아래 if문에 진입합니다.
        if keyword:
            # 최근 검색어(tbl_recent_search) 테이블에서 get_or_create()를 사용하여 최근 검색어를 저장합니다.
            recent_search, created = RecentSearch.enabled_objects.get_or_create(member_id=request.session.get('member').get('id'), keyword=keyword)

            # 만약 created가 False라면, 이미 테이블에 존재하지만 status가 0이라는 뜻입니다.
            if not created:
                # 따라서 status를 1로 변경한 후 save()를 통해 update합니다.
                recent_search.status = 1
                recent_search.updated_date = timezone.now()
                recent_search.save(update_fields=['status', 'updated_date'])

        # 선택된 카테고리와 검색어, 카테고리 목록 및 지역 목록을 context에 담아줍니다.
        # 카테고리와 검색어는 각각 없을 경우 None과 ''(빈 문자열)이 담기게 됩니다.
        context = {
            'selected_category': selected_category,
            'categories': categories,
            'regions': regions,
            'keyword': keyword
        }

        return render(request, 'activity/web/activity-web.html', context=context)


class ActivityListAPI(APIView):
    # 활동 목록 페이지에서 기간/모집종료여부/지역/카테고리 등의 필터 및 페이지네이션과 함께 fetch로 요청한 데이터를
    # 응답하는 REST API입니다.
    def post(self, request):
        # fetch 요청의 body에서 JSON.stringify()를 통해 전달받은 데이터입니다.
        data = request.data

        # 현재 로그인한 사용자의 id를 세션에서 받아옵니다.
        member_id = request.session.get('member').get('id')

        # 검색을 통해 활동 목록 페이지로 이동한 후 fetch 요청으로 검색어가 들어왔을 경우입니다.
        keyword = data.get('keyword', '')
        if keyword == "None":
            keyword = ""

        # 전달받은 페이지를 정수로 형변환하며, 기본값을 1로 설정해줍니다.
        page = int(data.get('page', 1))

        # 전달받은 기간 필터의 값을 담아주며, 기본값은 '모든날'입니다.
        date = data.get('date', '모든날')

        # 전달받은 지역 필터의 값을 담아주며, 기본값은 빈 문자열입니다.
        region = data.get('region', '')

        # 전달받은 카테고리 필터 리스트를 담아주며, 기본값은 빈 리스트입니다.
        categories = data.get('categories', [])

        # 전달받은 "모집 종료 활동 표시 여부"를 담아주며, 기본값은 False입니다.
        show_finished = data.get('showFinished', False)

        # 전달받은 정렬 옵션을 담아주며, 기본값은 '새 행사순(최신순)'입니다.
        ordering = data.get('ordering', '새 행사순')

        # 받아온 정렬 옵션을 바로 order_by()에서 일괄처리하기 위해 정렬 옵션에 따른 실제 입력값을 딕셔너리로 만들어줍니다.
        order_options = {
            '추천순': '-id',
            '새 행사순': '-id',
            '모집 마감일순': 'recruit_end'
        }

        # 페이지네이션을 위한 내용입니다.
        # 활동 목록에서는 한 페이지에 최대 12개의 활동을 표시합니다.
        row_count = 12

        # 현재 페이지의 시작 인덱스를 구합니다.
        offset = (page - 1) * row_count

        # 현재 페이지의 마지막 인덱스를 구합니다.
        limit = page * row_count

        # Queryset의 filter()에 추가할 조건을 초기화해줍니다.
        condition = Q()

        # 받아온 keyword의 포함여부를 조건에 추가합니다.
        condition &= Q(activity_title__icontains=keyword) | Q(activity_content__icontains=keyword)

        # 받아온 지역에 해당하는지 여부를 조건에 추가합니다.
        condition &= Q(activity_address_location__icontains=region)

        # 받아온 기간 필터에 따라 조건을 추가합니다.
        if date == '오늘':
            condition &= Q(activity_start__lte=timezone.now(), activity_end__gte=timezone.now())
        elif date == '이번주':
            condition &= Q(activity_start__lte=timezone.now() + timezone.timedelta(days=7),
                           activity_end__gte=timezone.now())
        elif date == '이번달':
            condition &= Q(activity_start__lte=timezone.now() + timezone.timedelta(weeks=4),
                           activity_end__gte=timezone.now())
        elif date == '모든날':
            condition = condition
        # 아래 else에 진입하는 경우는 기간 필터가 "기간 선택"일 경우입니다.
        # 화면에서 datetimepicker api를 통해 입력한 기간을 문자열로 전달받은 후 make_datetime() 함수를 통해 datetime객체로 변환합니다.
        else:
            start_date, end_date = date.split(' - ')
            start_date = make_datetime(start_date)
            end_date = make_datetime(end_date)
            condition &= Q(activity_start__lte=end_date, activity_end__gte=start_date)

        # 카테고리가 빈 리스트가 아닐 경우, 즉 카테고리 필터를 선택했을 경우 해당 리스트의 각 카테고리에 대해
        # 해당하는 활동들을 모두 보여주도록 __in을 사용하여 조건에 추가합니다.
        if categories:
            condition &= Q(category_id__in=categories)

        # 모집 종료 활동 표시 여부가 False라면 조건을 추가합니다.
        if not show_finished:
            condition &= Q(recruit_start__lte=timezone.now(), recruit_end__gte=timezone.now())

        # 활동 목록 페이지에서 페이지네이션과 관계없이 필터에 맞는 활동의 총 개수를 출력하기 위해 total_count에 담아줍니다.
        total_count = Activity.enabled_objects.filter(condition).count()

        # 한 화면에 보여줄 페이지 버튼의 개수를 5로 설정합니다.
        page_count = 5

        # 현재 페이지를 해당 개수로 나눈 후 math.ceil()을 통해 올림하여 다시 해당 개수를 곱함으로써 현재 보여줄 마지막 페이지를 정합니다.
        end_page = math.ceil(page / page_count) * page_count

        # 마지막 페이지로부터 현재 페이지네이션에 보여줄 첫 번째 페이지를 정합니다.
        start_page = end_page - page_count + 1

        # 실제 총 페이지의 개수를 구합니다.
        # 전체 활동 개수에서 한 페이지당 표시할 활동의 개수를 나눠준 후, 나머지가 생길 경우를 대비해 math.ceil()로 올려줍니다.
        real_end = math.ceil(total_count / row_count)

        # 이때 앞서 구한 현재 화면에서의 마지막 페이지가 더 크다면 마지막 페이지를 real_end로 바꿔줍니다.
        end_page = real_end if end_page > real_end else end_page

        # 게시물이 하나도 없어 페이지가 0이라면 화면에서 1로 표시하기 위해 1로 바꿔줍니다.
        if end_page == 0:
            end_page = 1

        # 화면 쪽 javascript에서 활용하기 위해 페이지네이션 관련 정보들을 page_info 딕셔너리에 담아줍니다.
        page_info = {
            'totalCount': total_count,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        # 모든 필터들 및 정렬 옵션을 적용한 활동들을 values()를 통해 딕셔너리로 변환한 후 페이지네이션에 맞추어 끊어줍니다.
        # 그리고 list타입으로 형변환합니다.
        activities = list(Activity.enabled_objects.filter(condition)\
                          .values().order_by(order_options[ordering])[offset:limit])

        # 각 활동마다 참가자 수 및 현재 로그인한 사용자가 관심활동으로 등록했는지 여부를 표시하기 위해 따로 조회하여
        # 딕셔너리에 key:value 형식으로 추가합니다.
        for activity in activities:
            activity['member_count'] = ActivityMember.enabled_objects.filter(activity_id=activity['id']).count()
            activity['is_like'] = ActivityLike.enabled_objects.filter(activity_id=activity['id'], member_id=member_id).exists()

        # 하나의 리스트로 응답하기 위해 리스트에 page_info와 total_count를 append()해줍니다.
        activities.append(page_info)
        activities.append(total_count)

        return Response(activities)


class ActivityCategoryAPI(APIView):
    # 헤더에서 카테고리 목록을 불러오기 위한 REST API입니다.
    def get(self, request):
        categories = list(Category.objects.filter(status=True).values())

        return Response(categories)


class ActivityJoinWebView(View):
    # 활동 신청 페이지로 이동하기 위한 view입니다.
    def get(self, request):
        # 쿼리스트링으로 활동의 id를 받아와 저장합니다.
        activity_id = request.GET.get('id')

        # 해당 id로 활동 객체를 조회합니다.
        activity = Activity.enabled_objects.get(id=activity_id)

        # 해당 활동의 구성원 수를 집계하여 담아줍니다.
        member_count = ActivityMember.enabled_objects.filter(activity_id=activity_id).count()

        # 화면에 표시하기 위한 해당 정보들을 context에 담아줍니다.
        context = {
            'activity': activity,
            'member_count': member_count
        }

        return render(request, 'activity/web/activity-join-web.html', context=context)

    # 활동 신청 요청에 응답하는 view입니다.
    def post(self, request):
        # post방식으로 받아온 데이터를 담아줍니다.
        data = request.POST

        # 참가 상태(status)는 기본값인 -1(승인 대기중)로 지정한 후 data 딕셔너리에 담아줍니다.
        data = {
            'activity_id': data.get('activity-id'),
            'member_id': data.get('member-id'),
            'status': -1
        }

        # get_or_create()를 통해 tbl_activity_member에 insert혹은 update해줍니다.
        activity_member, created = ActivityMember.objects.get_or_create(**data)

        # created가 False라면 이미 존재하는 데이터이며,
        # 이때 soft delete 혹은 신청 반려로 인해 status가 0일 경우에만 if문에 진입합니다.
        if not created and activity_member.status == 0:
            # status를 -1(승인 대기중)으로 변경한 후 save()를 통해 update합니다.
            activity_member.status = -1
            activity_member.updated_date = timezone.now()
            activity_member.save(update_fields=['status', 'updated_date'])

        # 활동 가입 신청에 대한 알림을 해당 활동을 개설한 모임의 모임장에게 전송합니다.
        # 받아온 활동의 id로 활동 객체를 조회합니다.
        activity = Activity.enabled_objects.filter(id=request.POST.get('activity-id')).first()

        # 활동 객체의 필드인 모임 객체의 id로 다시 모임 객체를 조회합니다.
        club = Club.enabled_objects.filter(id=activity.club.id).first()

        # 두 객체가 모두 None이 아닐 경우, 즉 유효할 경우 아래 if문에 진입하여 알림을 생성합니다.
        if activity and club:
            # 알림을 생성하기 위한 데이터를 딕셔너리에 담아준 후 create()를 통해 insert합니다.
            alarm_data = {
                'target_id': activity.id,
                'alarm_type': 11,
                'sender_id': request.POST.get('member-id'),
                'receiver_id': club.member.id
            }
            Alarm.objects.create(**alarm_data)

        # 활동 신청 후에는 마이페이지-나의 활동 페이지로 redirect합니다.
        return redirect('/member/mypage-activity/')


class ActivityImageUploadAPI(APIView):
    # 활동 개설 시 summernote editor에서 업로드한 이미지를 따로 처리하기 위한 REST API입니다.
    def post(self, request):
        # summernote editor에서 업로드한 이미지를 받아옵니다.
        upload_image = request.FILES.get('image')

        # 해당 이미지를 ActivityImage의 ImageField 타입 필드로 tbl_activity_image 테이블에 insert합니다.
        activity_image = ActivityImage.objects.create(image_path=upload_image)

        # 이때 status를 0으로 함으로써 사용자가 이미지 업로드 후 활동 개설을 완료하지 않았을 때
        # django-cron 라이브러리를 활용하여 Schedule()에 등록함으로써(tasks.py)
        # 매일 밤 12시마다 하루 전 날 자정까지의 tbl_activity_image에서 status가 0인 이미지의 경로에 맞는 파일을 삭제합니다.
        activity_image.status = 0
        activity_image.save(update_fields=['status'])

        # 화면에서 실제 저장된 이미지 경로를 받아 summernote editor에 바로 표시하기 위해 전달합니다.
        image_path = activity_image.image_path.url

        # 해당 이미지의 id를 화면으로 전달함으로써 활동 개설 완료 시 업로드한 이미지들의 id를 통해
        # 해당 컬럼들의 activity_id(fk)를 개설 완료한 활동의 id로 변경할 수 있도록 합니다.
        image_id = activity_image.id

        return Response({
            'image_path': image_path,
            'image_id': image_id
        })
