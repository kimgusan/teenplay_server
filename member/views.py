import math
import os
from pathlib import Path

import joblib
from bootpay_backend import BootpayBackend
from django.db import transaction
from django.db.models import F, Q, Value, Count
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views import View
from rest_framework.response import Response
from rest_framework.views import APIView

from activity.models import ActivityReply, Activity, ActivityLike, ActivityMember
from activity.views import make_datetime
from alarm.models import Alarm
from club.models import ClubPostReply, ClubMember, Club, ClubNotice, ClubCategory
from friend.models import Friend
from letter.models import Letter, ReceivedLetter, SentLetter
from member.models import Member, MemberFavoriteCategory, MemberProfile, MemberDeleteReason
from member.serializers import MemberSerializer
from notice.models import Notice
from pay.models import Pay, PayCancel
from teenplay_server.category import Category
from teenplay_server.models import Region
from wishlist.models import WishlistReply, Wishlist


class MemberLoginWebView(View):
    # 로그인 페이지로 이동하는 view입니다.
    def get(self, request):
        return render(request, 'member/web/login-web.html')


class MemberJoinWebView(View):
    # 회원가입 페이지로 이동하는 view입니다.
    def get(self, request):
        # 요청 경로에서 쿼리스트링으로 type을 받아옵니다.
        # google, kakao, naver 중 하나가 담기게 됩니다.
        member_type = request.GET['type']

        # 요청 경로에서 쿼리스트링으로 이메일 주소를 받아옵니다.
        member_email = request.GET['email']

        # 요청 경로에서 쿼리스트링으로 이름을 받아옵니다.
        member_nickname = request.GET['name']

        # 해당 정보들을 화면에 전달하기 위해 context에 담아줍니다.
        context = {
            'member_type': member_type,
            'member_email': member_email,
            'member_nickname': member_nickname
        }

        return render(request, 'member/web/join-web.html', context=context)

    # 회원가입 완료 시 데이터를 post방식으로 전달할 때 호출되는 메소드입니다.
    @transaction.atomic
    def post(self, request):
        # post방식으로 받아온 데이터를 담아줍니다.
        data = request.POST

        # 마케팅 정보 수신 동의 여부 checkbox를 getlist로 받아와 len()을 통해 판단합니다.
        marketing_agree = data.getlist('marketing_agree')
        # 길이가 0일 경우 미동의, 1일 경우 체크된 상태이므로 동의입니다.
        marketing_agree = True if len(marketing_agree) else False

        # 개인정보 수집, 이용 동의 여부 checkbox를 getlist로 받아와 len()을 통해 판단합니다.
        privacy_agree = data.getlist('privacy_agree')
        # 동의, 미동의 여부는 마찬가지 방식으로 결정합니다.
        privacy_agree = True if len(privacy_agree) else False

        # 받아온 데이터들을 data에 딕셔너리 형태로 저장합니다.
        data = {
            'member_email': data['member-email'],
            'member_nickname': data['member-name'],
            'member_marketing_agree': marketing_agree,
            'member_privacy_agree': privacy_agree,
            'member_type': data['member-type'],
            'member_phone': data['member-phone'],
            'member_keyword1': ' ',
            'member_keyword2': ' ',
            'member_keyword3': ' ',
            'member_address': ' '
        }

        # data 딕셔너리를 이용하여 tbl_member에 insert하고 반환된 객체를 member에 담아줍니다.
        member = Member.objects.create(**data)

        # 회원별 ai 모델을 생성합니다.
        # 1. 활동 추천 ai 모델
        model_path = os.path.join(Path(__file__).resolve().parent.parent, 'ai/ai/activity_recommender.pkl')
        model = joblib.load(model_path)
        member_model_path = f'ai/2024/05/20/activity_model{member.id}.pkl'
        os.makedirs(os.path.dirname(member_model_path), exist_ok=True)
        joblib.dump(model, member_model_path)
        member.member_recommended_activity_model = member_model_path
        member.save(update_fields=['member_recommended_activity_model'])

        # member에 담긴 객체를 직렬화하여 member에 다시 담아줍니다.
        member = MemberSerializer(member).data

        # 직렬화된 객체를 세션에 'member' 키값으로 저장합니다. (로그인)
        request.session['member'] = member

        # 메인페이지로 redirect 합니다.
        return redirect('/')

###########################################################
from django.http import HttpResponse


# 자체작업 합의 후 push해야합니다.
class MypageInfoWebView(View):
    def get(self, request):
        member_id = request.session.get('member')
        if member_id is None:
            return redirect('member:login')
        else:
            member = request.session.get('member')
            member_file = MemberProfile.objects.filter(member_id=member['id'])
            member_files = list(member_file.values('profile_path').filter(status=1))
            if len(member_files) != 0:
                request.session['member_files'] = member_files
            member_file = request.session.get('member_files')

            category_session = MemberFavoriteCategory.objects.filter(member_id=member['id'])
            category_session = list(category_session.values('status', 'category_id'))
            if len(category_session) != 0:
                request.session['member_category'] = category_session
            member_category = request.session.get('member_category')

            categories = Category.objects.all()

            return render(request, 'mypage/web/my-info-web.html',
                          {'member': member, 'categories': categories, 'member_files': member_file,
                           'member_category': member_category})

    @transaction.atomic
    def post(self, request):
        # 세션에서 member의 값을 가져옴
        member_data = request.session.get('member', 'member_files')
        file = request.FILES

        if member_data:
            member_id = member_data['id']

            try:
                # member_id에 해당하는 Member 객체를 가져옴
                member = Member.objects.get(id=member_id)
            except Member.DoesNotExist:
                # 만약 member_id에 해당하는 Member 객체가 없을 경우 에러 처리
                return HttpResponse("Member matching query does not exist.", status=404)

            data = request.POST
            if data.get('member-nickname') != '':
                member.member_nickname = data['member-nickname']
            if data.get('member-phone') != '':
                member.member_phone = data['member-phone']
            member.member_gender = int(data['gender'])
            if data.get('member-marketing_agree') == '1':
                member.member_marketing_agree = True
            else:
                member.member_marketing_agree = False
            if data.get('member-privacy_agree') == '1':
                member.member_privacy_agree = True
            else:
                member.member_privacy_agree = False

            if 'member-age' in data and data['member-age'] != '':
                member.member_birth = data['member-age']

            selected_categories = request.POST.getlist('selected_categories')

            # 기존의 회원 관심 카테고리 가져오기
            member_categories = MemberFavoriteCategory.objects.filter(member_id=member.id)

            # 선택된 각 카테고리에 대해 처리
            # MemberFavoriteCategory.objects.filter(member_id=member.id).update(status=0)
            for category_id in selected_categories:
                # 해당 카테고리가 이미 존재하는지 확인
                category_exists = member_categories.filter(category_id=category_id).exists()

                if category_exists:
                    # 이미 존재하면 status 변경
                    target_member_category = member_categories.filter(category_id=category_id).first()

                    if target_member_category:
                        # 특정 category_id에 대한 상태를 토글
                        target_member_category.status = 1 - target_member_category.status
                        target_member_category.save()
                else:
                    # 존재하지 않으면 새로운 레코드 생성
                    MemberFavoriteCategory.objects.create(member_id=member.id, category_id=category_id, status=1)

            # 관심 키워드 업데이트
            member.member_keyword1 = data.get('keyword1')
            member.member_keyword2 = data.get('keyword2')
            member.member_keyword3 = data.get('keyword3')

            member.updated_date = timezone.now()
            member.save(update_fields=['member_nickname', 'member_phone', 'member_gender', 'member_marketing_agree',
                                       'member_privacy_agree', 'member_birth', 'member_keyword1',
                                       'member_keyword2', 'member_keyword3', 'updated_date'])
            member_file = MemberProfile.objects.filter(member_id=member.id)
            for key, uploaded_file in file.items():
                try:
                    member_profile = MemberProfile.objects.get(member_id=member.id)
                    # 이미 프로필이 존재하는 경우 업데이트
                    member_profile.profile_path = uploaded_file
                    member_profile.save()
                except MemberProfile.DoesNotExist:
                    # 멤버 프로필이 없는 경우 생성
                    MemberProfile.objects.create(member_id=member.id, profile_path=uploaded_file, )

            # 수정된 데이터를 세션에 다시 저장
            request.session['member'] = {
                'id': member.id,
                'member_birth': member.member_birth,
                'member_email': member.member_email,
                'member_nickname': member.member_nickname,
                'member_address': member.member_address,
                'member_phone': member.member_phone,
                'member_gender': member.member_gender,
                'member_marketing_agree': member.member_marketing_agree,
                'member_privacy_agree': member.member_privacy_agree,
                'member_keyword1': member.member_keyword1,
                'member_keyword2': member.member_keyword2,
                'member_keyword3': member.member_keyword3,
            }
            member_files = list(member_file.values('profile_path').filter(status=1))
            if len(member_files) != 0:
                request.session['member_files'] = member_files

            category_session = MemberFavoriteCategory.objects.filter(member_id=member.id)
            category_session = list(category_session.values('status', 'category_id'))
            if len(category_session) != 0:
                request.session['member_category'] = category_session

            return redirect('member:mypage-info')
        else:
            # 세션 데이터에서 member가 없는 경우 에러 처리
            return HttpResponse("Session data for member is missing.", status=400)


class MypageDeleteWebView(View):
    def get(self, request):
        member = request.session['member']['id']
        frofile = MemberProfile.objects.filter(member_id=member)
        context = {'frofile': frofile

                   }
        return render(request, 'mypage/web/withdrawal-web.html', context)

    @transaction.atomic
    def post(self, request):
        member_data = request.session.get('member')
        text = request.POST.get("reason-text")
        number = request.POST.get("select_bum")
        MemberDeleteReason.objects.create(delete_text=text, delete_reason=number)
        member_id = member_data['id']
        member = Member.objects.get(id=member_id)
        member.status = 0
        member.updated_date = timezone.now()
        member.save(update_fields=['status', 'updated_date'])
        request.session.clear()
        return redirect('/')


class MypageLetterWebView(View):
    def get(self, request):
        member = request.session.get('member')
        latest_letter = Letter.objects.latest('id')  # 또는 필요한 기준으로 필터링
        return render(request, 'mypage/web/my-letter-web.html', {"member": member, 'letter_id': latest_letter.id})


class MypageWriteAPIWebView(APIView):
    @transaction.atomic
    def post(self, request):
        data = request.data

        receiver_email = data['receiver_id'].split("(")[1][:-1]
        receiver_instance = Member.objects.filter(member_email=receiver_email).first()
        receiver_id = receiver_instance.id
        # Letter 모델 생성
        letter = Letter.objects.create(
            receiver_id=receiver_id,
            letter_content=data['letter_content'],
            sender_id=request.session['member']['id']
        )

        # ReceivedLetter 모델 생성
        ReceivedLetter.objects.create(letter_id=letter.id)

        # SentLetter 모델 생성
        SentLetter.objects.create(letter_id=letter.id)
        Alarm.objects.create(target_id=letter.id, alarm_type=4, sender_id=letter.sender_id,
                             receiver_id=letter.receiver_id)

        return Response("good")


class MypageListAPIWebView(APIView):
    @transaction.atomic
    def get(self, request, member_id, page):
        status_letter = request.GET.get('status_letter')
        row_count = 10
        offset = (page - 1) * row_count
        limit = page * row_count

        # 전체 데이터 수를 세는 쿼리
        total_count = Letter.objects.filter(
            Q(sender_id=member_id, status=1) | Q(receiver_id=member_id, status=1)).count()

        # 전체 페이지 수 계산
        total_pages = (total_count + row_count - 1) // row_count

        # 페이지 범위에 해당하는 데이터 조회
        replies = Letter.objects.filter(Q(sender_id=member_id, status=1) | Q(receiver_id=member_id, status=1)) \
                      .values('letter_content', 'created_date', 'sender__member_nickname', 'receiver__member_nickname',
                              'id')[offset:limit]

        sender = Letter.objects.filter(sender_id=member_id, status=1) \
                     .values('letter_content', 'created_date', 'sender__member_nickname', 'receiver__member_nickname',
                             'id')[offset:limit]

        receiver = Letter.objects.filter(receiver_id=member_id, status=1) \
                       .values('letter_content', 'created_date', 'sender__member_nickname', 'receiver__member_nickname',
                               'id')[offset:limit]

        # 응답에 전체 페이지 수와 데이터 추가
        response_data = {
            'total_pages': total_pages,
            'replies': replies,

        }
        if status_letter == 'sender':
            response_data['replies'] = Letter.objects.filter(sender_id=member_id, status=1) \
                                           .values('letter_content', 'created_date', 'sender__member_nickname',
                                                   'receiver__member_nickname',
                                                   'id')[offset:limit]

            response_data['total_pages'] = (Letter.objects.filter(sender_id=member_id,
                                                                  status=1).count() + row_count - 1) // row_count
        elif status_letter == 'receiver':
            response_data['replies'] = Letter.objects.filter(receiver_id=member_id, status=1) \
                                           .values('letter_content', 'created_date', 'sender__member_nickname',
                                                   'receiver__member_nickname',
                                                   'id')[offset:limit]
            response_data['total_pages'] = (Letter.objects.filter(receiver_id=member_id,
                                                                  status=1).count() + row_count - 1) // row_count

        else:
            response_data['replies'] = Letter.objects.filter(
                Q(sender_id=member_id, status=1) | Q(receiver_id=member_id, status=1)) \
                                           .values('letter_content', 'created_date', 'sender__member_nickname',
                                                   'receiver__member_nickname', 'id')[offset:limit]

            response_data['total_pages'] = (Letter.objects.filter(
                Q(sender_id=member_id, status=1) | Q(receiver_id=member_id,
                                                     status=1)).count() + row_count - 1) // row_count

        return Response(response_data)


class MypageDeleteAPIWebView(APIView):
    @transaction.atomic
    def delete(self, request, letter_id):
        letter = Letter.objects.filter(id=letter_id)
        letter.update(status=0,updated_date=timezone.now())
        SentLetter.objects.filter(letter_id=letter_id).update(status=0, updated_date=timezone.now())
        ReceivedLetter.objects.filter(letter_id=letter_id).update(status=0, updated_date=timezone.now())

        return Response('good')

    @transaction.atomic
    def patch(self, request, letter_id):
        ReceivedLetter.objects.filter(letter_id=letter_id).update(is_read=0,updated_date=timezone.now())

        return Response('good')


class MypageCheckAPIWebViewMa(APIView):
    @transaction.atomic
    def post(self, request):
        member_email = request.data.get('member_email')

        member = Member.enabled_objects.filter(member_email=member_email)
        if member.exists():
            member = member.first()
            data = {
                'message': f'{member.member_nickname} ({member.member_email})'
            }
        else:
            data = {
                'message': '존재하지 않는 이메일입니다.'
            }

        return Response(data)


class MypageAlramView(View):
    @transaction.atomic
    def get(self, request):
        return render(request, 'mypage/web/my-signal-web.html')


class MypageAlramAPIView(APIView):
    @transaction.atomic
    def get(self, request, member_id, page):
        row_count = 5
        offset = (page - 1) * row_count
        limit = page * row_count

        alram = Alarm.objects.filter(receiver_id=member_id, status=1,is_read=1).values('id', 'alarm_type', 'target_id','created_date', 'receiver_id').order_by('-created_date')[offset:limit]

        return Response(alram)


class MypageAlramDeleteAPIView(APIView):
    @transaction.atomic
    def delete(self, request, alram_id):
        Alarm.objects.filter(id=alram_id).update(is_read=0,updated_date=timezone.now())
        print(Alarm)

        return Response('good')


class MypageTeenchinview(View):
    def get(self, request):
        return render(request, 'mypage/web/my-teenchin-web.html')


class MypageTeenchinAPIview(APIView):
    def get(self, request, member_id, page):
        row_count = 2
        offset = (page - 1) * row_count
        limit = page * row_count

        teenchin = Friend.objects.filter(receiver_id=member_id).values('id', 'is_friend')[offset:limit]

        return Response(teenchin)


############################################
class MemberAlarmCountAPI(APIView):
    # 로그인한 상태에서 헤더에 알람 개수를 띄우기 위한 요청에 응답하는 REST API 입니다.
    def get(self, request):
        # 로그인한 사용자의 id를 받아옵니다.
        member_id = request.GET.get('member-id')

        # tbl_alarm에서 해당 사용자에게 전송된(receiver_id=member_id) 알람의 개수를 집계하여 저장합니다.
        alarm_count = Alarm.enabled_objects.filter(receiver_id=member_id).count()

        # 이를 Response에 담아 응답합니다.
        return Response(alarm_count)


#############################################
class MypageTeenchinAPIview(APIView):
    def get(self, request, member_id, page):
        status_letter = request.GET.get('status_teenchin')
        search_text = request.GET.get('search')[:-1]

        row_count = 9
        offset = (page - 1) * row_count
        limit = page * row_count

        teenchin = []
        teenchin += Friend.objects.filter(sender_id=member_id, is_friend=1 ).values('id', 'is_friend', 'sender_id', 'receiver_id', 'sender__member_nickname',
                               'receiver__member_nickname','receiver__memberprofile__profile_path',)


        teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=1 ).values('id', 'is_friend', 'sender_id', 'receiver_id', 'sender__member_nickname',
                               'receiver__member_nickname','sender__memberprofile__profile_path')
        teenchin += Friend.objects.filter(sender_id=member_id, is_friend=-1).values('id', 'is_friend', 'sender_id',
                                                                                   'receiver_id',
                                                                                   'sender__member_nickname',
                                                                                   'receiver__member_nickname',
                                                                                   'receiver__memberprofile__profile_path')




        teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=-1).values('id', 'is_friend', 'sender_id',
                                                                                   'receiver_id',
                                                                                   'sender__member_nickname',
                                                                                   'receiver__member_nickname',
                                                                                   'sender__memberprofile__profile_path')


        if status_letter == 'case-a/':
            teenchin = []
            teenchin += Friend.objects.filter(sender_id=member_id, is_friend=1).values('id', 'is_friend', 'sender_id',
                                                                                       'receiver_id',
                                                                                       'sender__member_nickname',
                                                                                       'receiver__member_nickname',
                                                                                       'receiver__memberprofile__profile_path', )

            for aactivity_count in teenchin:
                receiver_id = aactivity_count.get('receiver_id')
                manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                member_id=receiver_id).values().count()
                aactivity_count['aactivity_count'] = manager_favorite_categories_add

            for club_count in teenchin:
                receiver_id = club_count.get('receiver_id')
                manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                            member_id=receiver_id).values().count()
                club_count['club_count'] = manager_favorite_categories_add



            teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=1).values('id', 'is_friend', 'sender_id',
                                                                                         'receiver_id',
                                                                                         'sender__member_nickname',
                                                                                         'receiver__member_nickname',
                                                                                         'sender__memberprofile__profile_path')
            if search_text:
                teenchin = []
                teenchin += Friend.objects.filter(sender_id=member_id, is_friend=1, receiver__member_nickname__icontains=search_text).values('id', 'is_friend',
                                                                                           'sender_id',
                                                                                           'receiver_id',
                                                                                           'sender__member_nickname',
                                                                                           'receiver__member_nickname',
                                                                                           'receiver__memberprofile__profile_path', )

                for aactivity_count in teenchin:
                    receiver_id = aactivity_count.get('receiver_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=receiver_id).values().count()
                    aactivity_count['aactivity_count'] = manager_favorite_categories_add

                for club_count in teenchin:
                    receiver_id = club_count.get('receiver_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=receiver_id).values().count()
                    club_count['club_count'] = manager_favorite_categories_add




                teenchin += Friend.objects.filter(sender_id=member_id, is_friend=1, receiver__member_email__icontains=search_text).values('id', 'is_friend',
                                                                                           'sender_id',
                                                                                           'receiver_id',
                                                                                           'sender__member_nickname',
                                                                                           'receiver__member_nickname',
                                                                                           'receiver__memberprofile__profile_path', )

                for aactivity_count in teenchin:
                    receiver_id = aactivity_count.get('receiver_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=receiver_id).values().count()
                    aactivity_count['aactivity_count'] = manager_favorite_categories_add

                for club_count in teenchin:
                    receiver_id = club_count.get('receiver_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=receiver_id).values().count()
                    club_count['club_count'] = manager_favorite_categories_add




                teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=1,sender__member_email__icontains=search_text).values('id', 'is_friend',
                                                                                             'sender_id',
                                                                                             'receiver_id',
                                                                                             'sender__member_nickname',
                                                                                             'receiver__member_nickname',
                                                                                             'sender__memberprofile__profile_path')
                teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=1,sender__member_nickname__icontains=search_text).values('id', 'is_friend',
                                                                                             'sender_id',
                                                                                             'receiver_id',
                                                                                             'sender__member_nickname',
                                                                                             'receiver__member_nickname',
                                                                                             'sender__memberprofile__profile_path')


        elif status_letter == 'case-b/':
            teenchin = []
            teenchin += Friend.objects.filter(sender_id=member_id, is_friend=-1).values('id', 'is_friend', 'sender_id',
                                                                                        'receiver_id',
                                                                                        'sender__member_nickname',
                                                                                        'receiver__member_nickname',
                                                                                        'receiver__memberprofile__profile_path')

            for aactivity_count in teenchin:
                receiver_id = aactivity_count.get('receiver_id')
                manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                member_id=receiver_id).values().count()
                aactivity_count['aactivity_count'] = manager_favorite_categories_add

            for club_count in teenchin:
                receiver_id = club_count.get('receiver_id')
                manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                            member_id=receiver_id).values().count()
                club_count['club_count'] = manager_favorite_categories_add







            if search_text:
                teenchin = []
                teenchin += Friend.objects.filter(sender_id=member_id, is_friend=-1,receiver__member_nickname__icontains=search_text).values('id', 'is_friend',
                                                                                            'sender_id',
                                                                                            'receiver_id',
                                                                                            'sender__member_nickname',
                                                                                            'receiver__member_nickname',
                                                                                            'receiver__memberprofile__profile_path')

                for aactivity_count in teenchin:
                    receiver_id = aactivity_count.get('receiver_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=receiver_id).values().count()
                    aactivity_count['aactivity_count'] = manager_favorite_categories_add

                for club_count in teenchin:
                    receiver_id = club_count.get('receiver_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=receiver_id).values().count()
                    club_count['club_count'] = manager_favorite_categories_add





                teenchin += Friend.objects.filter(sender_id=member_id, is_friend=-1,receiver__member_email__icontains=search_text).values('id', 'is_friend',
                                                                                            'sender_id',
                                                                                            'receiver_id',
                                                                                            'sender__member_nickname',
                                                                                            'receiver__member_nickname',
                                                                                            'receiver__memberprofile__profile_path')

                for aactivity_count in teenchin:
                    receiver_id = aactivity_count.get('receiver_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=receiver_id).values().count()
                    aactivity_count['aactivity_count'] = manager_favorite_categories_add

                for club_count in teenchin:
                    receiver_id = club_count.get('receiver_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=receiver_id).values().count()
                    club_count['club_count'] = manager_favorite_categories_add





        elif status_letter == 'case-c/':
            teenchin = []
            teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=-1).values('id', 'is_friend',
                                                                                          'sender_id',
                                                                                          'receiver_id',
                                                                                          'sender__member_nickname',
                                                                                          'receiver__member_nickname',
                                                                                          'sender__memberprofile__profile_path')

            for aactivity_count in teenchin:
                sender_id = aactivity_count.get('sender_id')
                manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                member_id=sender_id).values().count()
                aactivity_count['aactivity_countss'] = manager_favorite_categories_add

            for club_count in teenchin:
                sender_id = club_count.get('sender_id')
                manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                            member_id=sender_id).values().count()
                club_count['club_countss'] = manager_favorite_categories_add


            if search_text:
                teenchin = []
                teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=-1,sender__member_email__icontains=search_text).values('id', 'is_friend',
                                                                                              'sender_id',
                                                                                              'receiver_id',
                                                                                              'sender__member_nickname',
                                                                                              'receiver__member_nickname',
                                                                                              'sender__memberprofile__profile_path')

                for aactivity_count in teenchin:
                    sender_id = aactivity_count.get('sender_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=sender_id).values().count()
                    aactivity_count['aactivity_countss'] = manager_favorite_categories_add

                for club_count in teenchin:
                    sender_id = club_count.get('sender_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=sender_id).values().count()
                    club_count['club_countss'] = manager_favorite_categories_add


                teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=-1 ,sender__member_nickname__icontains=search_text).values('id', 'is_friend',
                                                                                              'sender_id',
                                                                                              'receiver_id',
                                                                                              'sender__member_nickname',
                                                                                              'receiver__member_nickname',
                                                                                              'sender__memberprofile__profile_path')

                for aactivity_count in teenchin:
                    sender_id = aactivity_count.get('sender_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=sender_id).values().count()
                    aactivity_count['aactivity_countss'] = manager_favorite_categories_add

                for club_count in teenchin:
                    sender_id = club_count.get('sender_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=sender_id).values().count()
                    club_count['club_countss'] = manager_favorite_categories_add



        else:
            teenchin = []
            teenchin += Friend.objects.filter(sender_id=member_id, is_friend=1).values('id', 'is_friend', 'sender_id',
                                                                                       'receiver_id',
                                                                                       'sender__member_nickname',
                                                                                       'receiver__member_nickname',
                                                                                       'receiver__memberprofile__profile_path', )

            for aactivity_count in teenchin:
                receiver_id = aactivity_count.get('receiver_id')
                manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                member_id=receiver_id).values().count()
                aactivity_count['aactivity_count'] = manager_favorite_categories_add

            for club_count in teenchin:
                receiver_id = club_count.get('receiver_id')
                manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                            member_id=receiver_id).values().count()
                club_count['club_count'] = manager_favorite_categories_add



            teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=1).values('id', 'is_friend', 'sender_id',
                                                                                         'receiver_id',
                                                                                         'sender__member_nickname',
                                                                                         'receiver__member_nickname',
                                                                                         'sender__memberprofile__profile_path')

            for aactivity_count in teenchin:
                sender_id = aactivity_count.get('sender_id')
                manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                member_id=sender_id).values().count()
                aactivity_count['aactivity_countss'] = manager_favorite_categories_add

            for club_count in teenchin:
                sender_id = club_count.get('sender_id')
                manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                            member_id=sender_id).values().count()
                club_count['club_countss'] = manager_favorite_categories_add


            teenchin += Friend.objects.filter(sender_id=member_id, is_friend=-1).values('id', 'is_friend', 'sender_id',
                                                                                        'receiver_id',
                                                                                        'sender__member_nickname',
                                                                                        'receiver__member_nickname',
                                                                                        'receiver__memberprofile__profile_path')

            for aactivity_count in teenchin:
                receiver_id = aactivity_count.get('receiver_id')
                manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                member_id=receiver_id).values().count()
                aactivity_count['aactivity_count'] = manager_favorite_categories_add

            for club_count in teenchin:
                receiver_id = club_count.get('receiver_id')
                manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                            member_id=receiver_id).values().count()
                club_count['club_count'] = manager_favorite_categories_add



            teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=-1).values('id', 'is_friend',
                                                                                          'sender_id',
                                                                                          'receiver_id',
                                                                                          'sender__member_nickname',
                                                                                          'receiver__member_nickname',
                                                                                          'sender__memberprofile__profile_path')



            for aactivity_count in teenchin:
                sender_id = aactivity_count.get('sender_id')
                manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                member_id=sender_id).values().count()
                aactivity_count['aactivity_countss'] = manager_favorite_categories_add

            for club_count in teenchin:
                sender_id = club_count.get('sender_id')
                manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                            member_id=sender_id).values().count()
                club_count['club_countss'] = manager_favorite_categories_add

            if search_text:
                teenchin = []
                teenchin += Friend.objects.filter(sender_id=member_id, is_friend=1,receiver__member_nickname__icontains=search_text).values('id', 'is_friend',
                                                                                           'sender_id',
                                                                                           'receiver_id',
                                                                                           'sender__member_nickname',
                                                                                           'receiver__member_nickname',
                                                                                           'receiver__memberprofile__profile_path', )

                for aactivity_count in teenchin:
                    receiver_id = aactivity_count.get('receiver_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=receiver_id).values().count()
                    aactivity_count['aactivity_count'] = manager_favorite_categories_add

                for club_count in teenchin:
                    receiver_id = club_count.get('receiver_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=receiver_id).values().count()
                    club_count['club_count'] = manager_favorite_categories_add




                teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=1 ,sender__member_nickname__icontains=search_text).values('id', 'is_friend',
                                                                                             'sender_id',
                                                                                             'receiver_id',
                                                                                             'sender__member_nickname',
                                                                                             'receiver__member_nickname',
                                                                                             'sender__memberprofile__profile_path')

                for aactivity_count in teenchin:
                    sender_id = aactivity_count.get('sender_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=sender_id).values().count()
                    aactivity_count['aactivity_countss'] = manager_favorite_categories_add

                for club_count in teenchin:
                    sender_id = club_count.get('sender_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=sender_id).values().count()
                    club_count['club_countss'] = manager_favorite_categories_add


                teenchin += Friend.objects.filter(sender_id=member_id, is_friend=-1,receiver__member_nickname__icontains=search_text).values('id', 'is_friend',
                                                                                            'sender_id',
                                                                                            'receiver_id',
                                                                                            'sender__member_nickname',
                                                                                            'receiver__member_nickname',
                                                                                            'receiver__memberprofile__profile_path')

                for aactivity_count in teenchin:
                    receiver_id = aactivity_count.get('receiver_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=receiver_id).values().count()
                    aactivity_count['aactivity_count'] = manager_favorite_categories_add

                for club_count in teenchin:
                    receiver_id = club_count.get('receiver_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=receiver_id).values().count()
                    club_count['club_count'] = manager_favorite_categories_add



                teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=-1,sender__member_nickname__icontains=search_text).values('id', 'is_friend',
                                                                                              'sender_id',
                                                                                              'receiver_id',
                                                                                              'sender__member_nickname',
                                                                                              'receiver__member_nickname',
                                                                                              'sender__memberprofile__profile_path')

                for aactivity_count in teenchin:
                    sender_id = aactivity_count.get('sender_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=sender_id).values().count()
                    aactivity_count['aactivity_countss'] = manager_favorite_categories_add

                for club_count in teenchin:
                    sender_id = club_count.get('sender_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=sender_id).values().count()
                    club_count['club_countss'] = manager_favorite_categories_add


                teenchin += Friend.objects.filter(sender_id=member_id, is_friend=1,receiver__member_email__icontains=search_text).values('id', 'is_friend',
                                                                                           'sender_id',
                                                                                           'receiver_id',
                                                                                           'sender__member_nickname',
                                                                                           'receiver__member_nickname',
                                                                                           'receiver__memberprofile__profile_path', )

                for aactivity_count in teenchin:
                    receiver_id = aactivity_count.get('receiver_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=receiver_id).values().count()
                    aactivity_count['aactivity_count'] = manager_favorite_categories_add

                for club_count in teenchin:
                    receiver_id = club_count.get('receiver_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=receiver_id).values().count()
                    club_count['club_count'] = manager_favorite_categories_add


                teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=1,sender__member_email__icontains=search_text).values('id', 'is_friend',
                                                                                             'sender_id',
                                                                                             'receiver_id',
                                                                                             'sender__member_nickname',
                                                                                             'receiver__member_nickname',
                                                                                             'sender__memberprofile__profile_path')

                for aactivity_count in teenchin:
                    sender_id = aactivity_count.get('sender_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=sender_id).values().count()
                    aactivity_count['aactivity_countss'] = manager_favorite_categories_add

                for club_count in teenchin:
                    sender_id = club_count.get('sender_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=sender_id).values().count()
                    club_count['club_countss'] = manager_favorite_categories_add


                teenchin += Friend.objects.filter(sender_id=member_id, is_friend=-1,receiver__member_email__icontains=search_text).values('id', 'is_friend',
                                                                                            'sender_id',
                                                                                            'receiver_id',
                                                                                            'sender__member_nickname',
                                                                                            'receiver__member_nickname',
                                                                                            'receiver__memberprofile__profile_path')

                for aactivity_count in teenchin:
                    receiver_id = aactivity_count.get('receiver_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=receiver_id).values().count()
                    aactivity_count['aactivity_count'] = manager_favorite_categories_add

                for club_count in teenchin:
                    receiver_id = club_count.get('receiver_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=receiver_id).values().count()
                    club_count['club_count'] = manager_favorite_categories_add


                teenchin += Friend.objects.filter(receiver_id=member_id, is_friend=-1,sender__member_email__icontains=search_text).values('id', 'is_friend',
                                                                                              'sender_id',
                                                                                              'receiver_id',
                                                                                              'sender__member_nickname',
                                                                                              'receiver__member_nickname',
                                                                                              'sender__memberprofile__profile_path')

                for aactivity_count in teenchin:
                    sender_id = aactivity_count.get('sender_id')
                    manager_favorite_categories_add = ActivityMember.objects.filter(status=1,
                                                                                    member_id=sender_id).values().count()
                    aactivity_count['aactivity_countss'] = manager_favorite_categories_add

                for club_count in teenchin:
                    sender_id = club_count.get('sender_id')
                    manager_favorite_categories_add = ClubMember.objects.filter(status=1,
                                                                                member_id=sender_id).values().count()
                    club_count['club_countss'] = manager_favorite_categories_add



        return Response(teenchin[offset:limit])


class MypageTeenchindeleteview(APIView):
    @transaction.atomic
    def delete(self, request, friend_id):
        print(friend_id)
        if Friend.objects.filter(receiver_id=friend_id,sender_id=request.session['member']['id']):
            target_id = Friend.objects.filter(receiver_id=friend_id, sender_id=request.session['member']['id']).update(is_friend=0, updated_date=timezone.now())
            Alarm.objects.create(target_id=target_id, receiver_id=friend_id, alarm_type=15,
                                 sender_id=request.session['member']['id'])

        elif Friend.objects.filter(receiver_id=request.session['member']['id'],sender_id=friend_id):
            target_id = Friend.objects.filter(receiver_id=request.session['member']['id'], sender_id=friend_id).update(is_friend=0, updated_date=timezone.now())


            Alarm.objects.create(target_id=target_id, receiver_id=friend_id, alarm_type=15,
                                 sender_id=request.session['member']['id'])

        return Response('good')

    @transaction.atomic
    def patch(self, request, friend_id):

        if Friend.objects.filter(receiver_id=friend_id, sender_id=request.session['member']['id']):
            target_id = Friend.objects.filter(receiver_id=friend_id, sender_id=request.session['member']['id']).update(
                is_friend=1, updated_date=timezone.now())
            Alarm.objects.create(target_id=target_id, receiver_id=friend_id, alarm_type=14,
                                 sender_id=request.session['member']['id'])

        elif Friend.objects.filter(receiver_id=request.session['member']['id'], sender_id=friend_id):
            target_id = Friend.objects.filter(receiver_id=request.session['member']['id'], sender_id=friend_id).update(
                is_friend=1, updated_date=timezone.now())

            Alarm.objects.create(target_id=target_id, receiver_id=friend_id, alarm_type=14,
                                 sender_id=request.session['member']['id'])

        return Response('good')


class MypageTeenchinLetterAPIview(APIView):
    @transaction.atomic
    def post(self, request):
        data = request.data

        receiver_id = data['receiver_id']
        letter = Letter.objects.create(
            receiver_id=receiver_id,
            letter_content=data['letter_content'],
            sender_id=request.session['member']['id']
        )
        ReceivedLetter.objects.create(letter_id=letter.id)

        # SentLetter 모델 생성
        SentLetter.objects.create(letter_id=letter.id)
        Alarm.objects.create(target_id=letter.id, alarm_type=4, sender_id=letter.sender_id,
                             receiver_id=letter.receiver_id)

        return Response("good")


class MapagePaymentView(View):
    def get(self, request):
        return render(request, 'mypage/web/my-payment-web.html')


class MypagePayListAPIVIEW(APIView):

    def get(self, request, member_id, page):
        row_count = 3
        offset = (page - 1) * row_count
        limit = page * row_count

        pay = Pay.objects.filter(member_id=member_id, status=1).values('id', 'created_date', 'activity__activity_title',
                                                                       'activity__club__club_name',
                                                                       'activity__activity_intro',
                                                                       'member__club__club_intro',
                                                                       'activity__activity_title',
                                                                       'activity__thumbnail_path',
                                                                       'activity__id',
                                                                       'activity__recruit_end')[offset:limit]




        return Response(pay)


class MypagePayDeleteAPIVIEW(APIView):
    @transaction.atomic
    def post(self, request):
        pay_reason = request.data
        reason = pay_reason['reason']
        pay_id = pay_reason['pay']
        activity = Activity.objects.filter(pay_id=pay_id).first()
        pay = Pay.objects.filter(id=pay_id).first()
        pay.status = 0
        pay.updated_date=timezone.now()
        pay.save(update_fields=['status','updated_date'])
        activity.status = 0
        activity.updated_date=timezone.now()
        activity.save(update_fields=['status','updated_date'])
        receipt_id = pay.receipt_id
        bootpay = BootpayBackend('65e44626e57a7e001be37370',
                                 'NQmDRBsgOfziMiNXUEKrJGQ+YhXZncneSVG/auKihFA=')
        token = bootpay.get_access_token()
        if 'error_code' not in token:
            bootpay.cancel_payment(receipt_id=receipt_id,
                                   cancel_price=pay.price,
                                   cancel_username='관리자', cancel_message='취소됨')

        PayCancel.objects.create(pay_cancel_reason=reason)


        return Response('good')


class MypageReplyView(View):
    def get(self, request):
        return render(request, 'mypage/web/my-reply-web.html')


class MypageReplyAPIVIEW(APIView):
    def get(self, request, member_id, page):
        status_reply = request.GET.get('status_reply')
        row_count = 10
        offset = (page - 1) * row_count
        limit = page * row_count
        print(status_reply)

        total_count = 0

        total_count += WishlistReply.objects.filter(member_id=member_id, status=1).count()
        total_count += ActivityReply.objects.filter(member_id=member_id, status=1).count()
        total_count += ClubPostReply.objects.filter(member_id=member_id, status=1).count()

        # 전체 페이지 수 계산
        total_pages = (total_count + row_count - 1) // row_count
        reply = []

        reply += WishlistReply.objects.filter(member_id=member_id, status=1).values('id', 'reply_content',
                                                                                    'created_date', 'wishlist_id',
                                                                                    'wishlist__wishlist_content')
        reply += ActivityReply.objects.filter(member_id=member_id, status=1).values('id', 'reply_content',
                                                                                    'created_date', 'activity_id',
                                                                                    'activity__activity_title')
        reply += ClubPostReply.objects.filter(member_id=member_id, status=1).values('id', 'reply_content',
                                                                                    'created_date', 'club_post_id',
                                                                                    'club_post__post_title')
        reply.sort(key=lambda x: x['created_date'], reverse=True)
        response_data = {
            'total_pages': total_pages,
            'reply': reply[offset:limit],

        }

        if status_reply == 'wish':
            response_data['reply'] = WishlistReply.objects.filter(member_id=member_id, status=1).values('id',
                                                                                                        'reply_content',
                                                                                                        'created_date',
                                                                                                        'wishlist_id',
                                                                                                        'wishlist__wishlist_content')[
                                     offset:limit]
            response_data['total_pages'] = (WishlistReply.objects.filter(member_id=member_id,
                                                                         status=1).count() + row_count - 1) // row_count
            print(response_data['total_pages'])

        elif status_reply == 'post':
            response_data['reply'] = ClubPostReply.objects.filter(member_id=member_id, status=1).values('id',
                                                                                                        'reply_content',
                                                                                                        'created_date',
                                                                                                        'club_post_id',
                                                                                                        'club_post__post_title')[
                                     offset:limit]
            response_data['total_pages'] = (ClubPostReply.objects.filter(member_id=member_id,
                                                                         status=1).count() + row_count - 1) // row_count

        elif status_reply == 'ac':
            response_data['reply'] = ActivityReply.objects.filter(member_id=member_id, status=1).values('id',
                                                                                                        'reply_content',
                                                                                                        'created_date',
                                                                                                        'activity_id',
                                                                                                        'activity__activity_title')[
                                     offset:limit]
            response_data['total_pages'] = (ActivityReply.objects.filter(member_id=member_id,
                                                                         status=1).count() + row_count - 1) // row_count

        else:

            total_count = 0

            total_count += WishlistReply.objects.filter(member_id=member_id, status=1).count()
            total_count += ActivityReply.objects.filter(member_id=member_id, status=1).count()
            total_count += ClubPostReply.objects.filter(member_id=member_id, status=1).count()

            # 전체 페이지 수 계산
            total_pages = (total_count + row_count - 1) // row_count
            print(total_pages)

            reply = []

            reply += WishlistReply.objects.filter(member_id=member_id, status=1).values('id', 'reply_content',
                                                                                        'created_date', 'wishlist_id',
                                                                                        'wishlist__wishlist_content')
            reply += ActivityReply.objects.filter(member_id=member_id, status=1).values('id', 'reply_content',
                                                                                        'created_date', 'activity_id',
                                                                                        'activity__activity_title')
            reply += ClubPostReply.objects.filter(member_id=member_id, status=1).values('id', 'reply_content',
                                                                                        'created_date', 'club_post_id',
                                                                                        'club_post__post_title')
            reply.sort(key=lambda x: x['created_date'], reverse=True)
            response_data = {
                'total_pages': total_pages,
                'reply': reply[offset:limit],

            }

        return Response(response_data)


class MypageReplyDeleteAPIVIEW(APIView):
    @transaction.atomic
    def delete(self, request, reply_id):
        if reply_id.startswith('a'):
            ActivityReply.objects.filter(id=int(reply_id[1:])).update(status=0,updated_date=timezone.now())
        elif reply_id.startswith('w'):
            WishlistReply.objects.filter(id=int(reply_id[1:])).update(status=0,updated_date=timezone.now())
        elif reply_id.startswith('p'):
            ClubPostReply.objects.filter(id=int(reply_id[1:])).update(status=0,updated_date=timezone.now())

        return Response('good')


############################################################
# 댓글창의 프로필 사진을 클릭하여 틴친(친구 서비스) 기능을 이용할 때 호출하는 REST API 입니다.
class TeenChinAPI(APIView):
    # 로그인한 사용자와 클릭한 사용자와의 틴친 관계를 조회할 때 호출되는 메소드입니다.
    def get(self, request):
        # 로그인한 사용자의 id를 세션에서 받아옵니다.
        member_id = request.session.get('member').get('id')

        # 요청 경로에서 쿼리스트링으로 상대방 id를 받아옵니다.
        teenchin_id = request.GET.get('teenchin-id')

        # 사용자가 신청을 보낸 사람(sender)일 수도 있고, 받은 사람(receiver)일 수도 있으므로 두 경우 모두 고려하여 조회합니다.
        send_friend = Friend.objects.filter(sender_id=member_id, receiver_id=teenchin_id)
        receive_friend = Friend.objects.filter(receiver_id=member_id, sender_id=teenchin_id)

        # 친구 상태에 해당하는 값을 0으로 초기화합니다.
        teenchin_status = 0

        # 화면의 javascript에서 이후 친구수락/친구거절/신청취소/친구끊기 등을 요청할 때 sender인지 여부를 판단하기 위해
        # 변수를 선언하고 False로 초기화합니다.
        is_sender = False

        # 만약 로그인한 사용자가 sender라면
        if send_friend.exists():
            # Friend 객체를 가져옵니다.
            send_friend = send_friend.first()
            # 친구 상태 변수에 해당 객체의 is_friend 필드값을 넣어줍니다.
            teenchin_status = send_friend.is_friend
            # 사용자가 sender이므로 is_sender에 True를 담아줍니다.
            is_sender = True

        # 만약 로그인한 사용자가 receiver라면
        elif receive_friend.exists():
            # Friend 객체를 가져옵니다.
            receive_friend = receive_friend.first()
            # 친구 상태 변수에 해당 객체의 is_friend 필드값을 넣어줍니다.
            teenchin_status = receive_friend.is_friend

        # if와 elif 조건문에 모두 들어가지 않았다면 둘 사이의 틴친 기록이 없다는 뜻이므로,
        # teenchin_status 및 is_sender가 각각 0, False로 초기화된 값이 들어가게 됩니다.
        return Response({
            'teenchinStatus': teenchin_status,
            'isSender': is_sender,
        })

    # 친구 신청을 보냈을 경우 호출되는 post 메소드입니다.
    @transaction.atomic
    def post(self, request):
        # 로그인한 사용자의 id를 세션에서 가져옵니다.
        member_id = request.session.get('member').get('id')

        # 상대방 id를 post로 전달받은 데이터에서 가져옵니다.
        teenchin_id = request.data.get('teenchinId')

        # 혹시 이미 틴친인데 post 요청이 들어왔을 경우 바로 return합니다.
        # 이때에도 사용자가 sender일 경우와 receiver일 경우 모두 고려하여 조회 및 판단합니다.
        send_friend = Friend.objects.filter(sender_id=member_id, receiver_id=teenchin_id, is_friend=1)
        if send_friend.exists():
            return Response("already exists")
        receive_friend = Friend.objects.filter(sender_id=teenchin_id, receiver_id=member_id, is_friend=1)
        if receive_friend.exists():
            return Response("already exists")

        # is_friend가 0인 컬럼이 있는지 먼저 조회합니다.
        # 이때 사용자가 sender일 경우와 receiver일 경우 모두 고려하여 조회합니다.
        # is_friend가 0이라면 soft delete (이 경우 틴친 끊기를 통해 끊어진 상태입니다) 된 상태입니다.
        send_friend = Friend.objects.filter(sender_id=member_id, receiver_id=teenchin_id, is_friend=0)
        if send_friend.exists():
            # 존재한다면 객체를 가져옵니다.
            send_friend = send_friend.first()
            # 객체의 is_friend 값을 -1(수락 대기중)로 변경한 후 save()를 통해 update합니다.
            send_friend.is_friend = -1
            send_friend.updated_date = timezone.now()
            send_friend.save(update_fields=['is_friend', 'updated_date'])

            # 요청을 받는 상대방에게 알람을 전송합니다.
            alarm_data = {
                'target_id': member_id,
                'alarm_type': 5,
                'sender_id': member_id,
                'receiver_id': teenchin_id
            }
            # 이때 get_or_create()를 통해 알람 객체를 생성하거나 가져옵니다.
            alarm, created = Alarm.objects.get_or_create(**alarm_data)

            # 만약 created가 False라면 soft delete된 경우이므로 status를 1로 다시 바꾼 후 save()합니다.
            if not created:
                alarm.status = 1
                alarm.updated_date = timezone.now()
                alarm.save(update_fields=['status', 'updated_date'])

            return Response("success")

        # 이번에는 사용자가 receiver일 경우에 해당합니다.
        receive_friend = Friend.objects.filter(sender_id=teenchin_id, receiver_id=member_id, is_friend=0)
        if receive_friend.exists():
            # 이미 존재한다면 객체를 가져옵니다.
            receive_friend = receive_friend.first()

            # 이번에는 receiver였던 사용자가 다시 요청을 보냈기 때문에,
            # sender와 receiver를 서로 바꿔줍니다.
            receive_friend.sender_id = member_id
            receive_friend.receiver_id = teenchin_id

            # 그리고 해당 객체의 is_friend를 -1(수락 대기중)으로 변경 후 save()를 통해 update합니다.
            receive_friend.is_friend = -1
            receive_friend.updated_date = timezone.now()
            receive_friend.save(update_fields=['is_friend', 'updated_date', 'sender_id', 'receiver_id'])

            # 틴친 신청을 받는 상대방에게 알람을 전송합니다.
            alarm_data = {
                'target_id': member_id,
                'alarm_type': 5,
                'sender_id': member_id,
                'receiver_id': teenchin_id
            }

            # get_or_create()를 통해 알람을 생성하거나 가져옵니다.
            alarm, created = Alarm.objects.get_or_create(**alarm_data)

            # 만약 created가 False라면 soft delete된 상태이므로 status를 1로 변경한 후 save()합니다.
            if not created:
                alarm.status = 1
                alarm.updated_date = timezone.now()
                alarm.save(update_fields=['status', 'updated_date'])

            return Response("success")

        # 만약 사용자가 sender와 receiver일 경우에 대해 모두 테이블에 정보가 존재하지 않는다면,
        # sender를 사용자로, receiver를 상대방으로 tbl_friend에 insert합니다.
        Friend.objects.create(sender_id=member_id, receiver_id=teenchin_id)

        # 마찬가지로 알람을 get_or_create()를 사용하여 생성하거나, 혹은 가져와서 status를 1로 변경 후 save()합니다.
        alarm_data = {
            'target_id': member_id,
            'alarm_type': 5,
            'sender_id': member_id,
            'receiver_id': teenchin_id
        }
        alarm, created = Alarm.objects.get_or_create(**alarm_data)
        if not created:
            alarm.status = 1
            alarm.updated_date = timezone.now()
            alarm.save(update_fields=['status', 'updated_date'])

        return Response("success")

    # 틴친 신청이 왔을 때 수락/거절 하거나, 내가 보낸 틴친 신청을 취소할 때 호출되는 patch 메소드입니다.
    @transaction.atomic
    def patch(self, request):
        # 로그인한 사용자의 id를 세션에서 가져옵니다.
        member_id = request.session.get('member').get('id')

        # fetch 요청의 body에서 상대방 id를 가져옵니다.
        teenchin_id = request.data.get('teenchinId')

        # fetch 요청의 body에서 사용자가 신청했는지 여부를 가져옵니다. 이때에는 신청을 취소하는 요청에 해당합니다.
        is_sender = request.data.get('isSender')

        # fetch 요청의 body에서, 사용자가 신청을 받았을 경우 수락/거절 여부를 가져옵니다.
        is_accept = request.data.get('isAccept')

        # is_sender > True -> 신청 취소만 가능 (is_accept는 관계없음)
        # is_sender > False -> is_accept가 True면 수락, False면 거절

        # 혹시 이미 틴친인데 요청이 들어왔을 경우 바로 return합니다.
        # 이때에도 사용자가 sender일 경우와 receiver일 경우 모두 고려하여 조회 및 판단합니다.
        send_friend = Friend.objects.filter(sender_id=member_id, receiver_id=teenchin_id, is_friend=1)
        if send_friend.exists():
            return Response("already exists")
        receive_friend = Friend.objects.filter(sender_id=teenchin_id, receiver_id=member_id, is_friend=1)
        if receive_friend.exists():
            return Response("already exists")

        # 사용자가 신청을 보냈을 경우입니다.
        if is_sender:
            # is_friend가 -1(수락 대기/신청중)인 정보를 tbl_friend에서 가져옵니다.
            send_friend = Friend.objects.filter(sender_id=member_id, receiver_id=teenchin_id, is_friend=-1)
            if send_friend.exists():
                # 존재할 경우 first()로 객체를 추출합니다.
                send_friend = send_friend.first()
                # is_friend를 0(거절/신청취소)으로 바꿔준 후 save()를 통해 update합니다.
                send_friend.is_friend = 0
                send_friend.updated_date = timezone.now()
                send_friend.save(update_fields=['is_friend', 'updated_date'])
                # 친구 신청 시 상대방에게 전송되었던 알람의 status를 0(soft deleted)으로 바꾼 후 save()합니다.
                alarm_data = {
                    'target_id': member_id,
                    'alarm_type': 5,
                    'sender_id': member_id,
                    'receiver_id': teenchin_id,
                }
                alarm = Alarm.objects.filter(**alarm_data)
                if alarm.exists():
                    alarm = alarm.first()
                    alarm.status = 0
                    alarm.updated_date = timezone.now()
                    alarm.save(update_fields=['status', 'updated_date'])

                return Response("success")

        # 사용자가 신청을 받았을 경우 수락/거절하는 부분입니다.
        # is_friend가 -1(수락 대기/신청중)인 정보를 tbl_friend에서 가져옵니다.
        receive_friend = Friend.objects.filter(sender_id=teenchin_id, receiver_id=member_id, is_friend=-1)
        if receive_friend.exists():
            # 존재할 경우 first()를 통해 객체를 추출합니다.
            receive_friend = receive_friend.first()
            # is_accept가 True라면 사용자가 '수락'을 선택했을 경우입니다.
            if is_accept:
                # 해당 객체의 is_friend를 1(틴친중)로 바꾼 후 save()하여 update합니다.
                receive_friend.is_friend = 1
                receive_friend.updated_date = timezone.now()
                receive_friend.save(update_fields=['is_friend', 'updated_date'])
                # 틴친 신청을 보냈던 상대방에게 틴친 승인 알림(alarm_type=14)을 보냅니다.
                alarm_data = {
                    'target_id': member_id,
                    'alarm_type': 14,
                    'sender_id': member_id,
                    'receiver_id': teenchin_id
                }
                # 이때 get_or_create()를 통해 알람 객체를 생성하거나 가져옵니다.
                alarm, created = Alarm.objects.get_or_create(**alarm_data)
                # created가 False라면, soft delete된 알람이므로 status를 1로 바꾼 후 save()합니다.
                if not created:
                    alarm.status = 1
                    alarm.updated_date = timezone.now()
                    alarm.save(update_fields=['status', 'updated_date'])

                return Response("success")

            # is_accept가 False라면 if문 블록에 진입하지 않고 이 부분이 실행됩니다.
            # is_friend를 0(거절/틴친끊기)으로 바꾼 후 save()합니다.
            receive_friend.is_friend = 0
            receive_friend.updated_date = timezone.now()
            receive_friend.save(update_fields=['is_friend', 'updated_date'])

            # 사용자에게 전송되었던 알림(alarm_type=5, 틴친신청 알림)이 있는지 조회합니다.
            alarm_data = {
                'target_id': teenchin_id,
                'alarm_type': 5,
                'sender_id': teenchin_id,
                'receiver_id': member_id,
            }
            alarm = Alarm.objects.filter(**alarm_data)

            # 해당 알림이 존재한다면,
            if alarm.exists():
                # first()를 통해 객체를 가져옵니다.
                alarm = alarm.first()
                # status를 0(soft delete)으로 바꾼 후 save()합니다.
                alarm.status = 0
                alarm.updated_date = timezone.now()
                alarm.save(update_fields=['status', 'updated_date'])

            # 사용자에게 틴친 신청을 보낸 상대방에게 틴친 거절 알림(alarm_type=15)을 전송합니다.
            alarm_data = {
                'target_id': member_id,
                'alarm_type': 15,
                'sender_id': member_id,
                'receiver_id': teenchin_id
            }
            # 이때 get_or_create()를 통해 알람 객체를 생성하거나 가져옵니다.
            alarm, created = Alarm.objects.get_or_create(**alarm_data)
            # created가 False라면 soft delete된 알람이므로 status를 1로 바꾼 후 save()합니다.
            if not created:
                alarm.status = 1
                alarm.updated_date = timezone.now()
                alarm.save(update_fields=['status', 'updated_date'])

            return Response("success")

        # 위 조건식이 모두 False였다면 아무런 작업도 하지 않고 return 합니다.
        return Response("fail")

    # 틴친 끊기 요청 시 호출되는 delete 메소드입니다.
    # 틴친을 끊었을 때에는 관련 알림이 전송되지 않습니다.
    @transaction.atomic
    def delete(self, request):
        # 현재 로그인한 사용자의 id를 세션에서 가져옵니다.
        member_id = request.session.get('member').get('id')

        # fetch 요청의 body에서 상대방의 id를 가져옵니다.
        teenchin_id = request.data.get('teenchinId')

        # 사용자와 상대방이 틴친중이라면, 사용자가 sender인 경우와 receiver인 경우를 모두 고려해주어야 합니다.
        # tbl_friend에서 사용자가 sender이고 is_friend가 1(틴친중)인 정보를 조회합니다.
        send_friend = Friend.objects.filter(sender_id=member_id, receiver_id=teenchin_id, is_friend=1)
        # 해당 정보가 존재한다면,
        if send_friend.exists():
            # first()를 통해 객체를 가져옵니다.
            send_friend = send_friend.first()
            # 해당 객체의 is_friend를 0(거절/틴친끊기)으로 바꾼 후 save()합니다.
            send_friend.is_friend = 0
            send_friend.updated_date = timezone.now()
            send_friend.save(update_fields=['is_friend', 'updated_date'])

            return Response("success")

        # 이번에는 사용자가 receiver이고 is_friend가 1(틴친중)인 정보를 조회합니다.
        receive_friend = Friend.objects.filter(sender_id=teenchin_id, receiver_id=member_id, is_friend=1)
        # 해당 정보가 존재한다면,
        if receive_friend.exists():
            # first()를 통해 객체를 가져옵니다.
            receive_friend = receive_friend.first()
            # 해당 객체의 is_friend를 0(거절/틴친끊기)으로 바꾼 후 save()합니다.
            receive_friend.is_friend = 0
            receive_friend.updated_date = timezone.now()
            receive_friend.save(update_fields=['is_friend', 'updated_date'])

            return Response("success")

        # 위 조건식이 모두 False라면 아무런 작업도 하지 않고 return합니다.
        return Response("fail")


class ClubAlarmManageAPI(APIView):
    # 모임 관련 알림을 각 모임마다 켜거나 끌 때 호출되는 REST API 입니다.
    def get(self, request):
        # 현재 로그인한 사용자의 id를 세션에서 가져옵니다.
        member_id = request.session.get('member').get('id')

        # fetch 요청 경로의 쿼리스트링에서 해당 모임의 id를 가져옵니다.
        club_id = request.GET.get('club-id')

        # 사용자id와 모임id로 tbl_club_member에서 사용자가 해당 모임의 구성원인지 조회합니다.
        club_member = ClubMember.enabled_objects.filter(member_id=member_id, club_id=club_id)

        # 모임에 대한 알림은 모임 구성원일 경우에만 켜거나 끌 수 있으므로 존재하지 않을 경우 return합니다.
        if not club_member.exists():
            return Response("Not found")

        # 존재할 경우 first()를 통해 객체를 가져옵니다.
        club_member = club_member.first()

        # alarm_status는 0(False)과 1(True) 둘 중의 하나이므로, not 연산자로 반대로 바꿔줍니다.
        # 바꿔준 후 save()를 통해 update합니다.
        club_member.alarm_status = not club_member.alarm_status
        club_member.updated_date = timezone.now()
        club_member.save(update_fields=['alarm_status', 'updated_date'])

        return Response("success")


################################################################################
class MypageActivityListView(View):
    def get(self, request, club_id):
        order = request.GET.get('order', '최신순')
        page = request.GET.get('page', 1)
        page_info = {
            'order': order,
            'page': page
        }

        club_profile = Club.enabled_objects.filter(id=club_id).values('club_profile_path').first()
        club_name = Club.enabled_objects.filter(id=club_id).values('club_name').first()
        print(club_profile)
        club_member_count = ClubMember.enabled_objects.filter(club_id=club_id).count()

        club_notice_queryset = ClubNotice.objects.filter(club_id=club_id)
        club_notice_count = club_notice_queryset.count()
        # club_notices = club_notice_queryset.values('id', 'notice_title', 'notice_content', 'status')

        waiting_member_count = ClubMember.objects.filter(club_id=club_id, status=-1).count()

        activity_queryset = Activity.objects.filter(
            club_id=club_id,
            recruit_start__lt=timezone.now(),
            recruit_end__gt=timezone.now(),
            status=1
        )
        activity_count = activity_queryset.count()

        activity_end_queryset = Activity.objects.filter(club_id=club_id, activity_end__lt=timezone.now(), status=1)
        activity_end_count = activity_end_queryset.count()

        insite = {
            'club_member_count': club_member_count,
            'club_notice_count': club_notice_count,
            'activity_count': activity_count,
            'waiting_member_count': waiting_member_count,
            'activity_end_count': activity_end_count,
            'club_profile': club_profile.get('club_profile_path') if club_profile else '',
            'club_name': club_name.get('club_name')
        }
        print(insite)
        activities_queryset = Activity.enabled_objects.filter(club_id=club_id, activity_end__gt=timezone.now(),
                                                              status=1)
        activities_count = activities_queryset.count()

        notices = Notice.objects.filter(notice_type=0).values('id', 'notice_title', 'created_date')[:4]
        for notice in notices:
            notice['created_date'] = notice['created_date'].strftime("%Y.%m.%d")

        club_notices = ClubNotice.objects.filter(club_id=club_id, status=1).values('id', 'notice_title',
                                                                                   'created_date')[:4]
        for club_notice in club_notices:
            club_notice['created_date'] = club_notice['created_date'].strftime("%Y.%m.%d")

        context = {
            'insite': insite,
            'page_info': page_info,
            'activities_count': activities_count,
            'notices': notices,
            'club_notices': club_notices,
            'club_id': club_id
        }

        return render(request, 'mypage/web/management-club-web.html', context)

    def post(self, request):
        datas = request.POST

        context = {
            'page_info': {
                'order': datas.get('order', '최신순'),
                'page': datas.get('page', 1)
            }
        }

        return render(request, 'mypage/web/management-club-web.html', context)


class MypageActivityListAPI(APIView):
    def post(self, request, club_id):
        order = request.data.get('order', '최신 개설순')
        page = int(request.data.get('page', 1))

        row_count = 6
        offset = (page - 1) * row_count
        limit = page * row_count

        weekday_translation = {
            'Mon': '월',
            'Tue': '화',
            'Wed': '수',
            'Thu': '목',
            'Fri': '금',
            'Sat': '토',
            'Sun': '일'
        }
        sort_field = '-created_date'
        if order == '최신 개설순':
            sort_field = '-created_date'
        elif order == '활동 일시순':
            sort_field = '-activity_start'

        # 조회수 어딧느닞 모르겠음
        # elif order == '인기순':
        #     pass

        activities_queryset = Activity.enabled_objects.filter(club_id=club_id, activity_end__gt=timezone.now(),
                                                              status=1)

        total_count = activities_queryset.count()

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

        activities = list(activities_queryset.values('id', 'activity_title', 'activity_start', 'created_date').order_by(
            sort_field)[offset:limit])

        for activity in activities:
            activity_start = activity['activity_start']
            create_date = activity['created_date']
            activity['activity_start'] = activity_start.strftime("%m월 %d일") + '(' + weekday_translation[
                activity_start.strftime('%a')] + ')' + activity_start.strftime(" %H:%M")
            activity['created_date'] = create_date.strftime("%m월 %d일") + '(' + weekday_translation[
                create_date.strftime('%a')] + ')' + create_date.strftime(" %H:%M")

        activities.append(page_info)
        return Response(activities)


class MypageMyClubView(View):
    def get(self, request):
        order = request.GET.get('order', '이름 순')
        page = request.GET.get('page', 1)

        context = {
            'order':order,
            'page':page
        }
        return render(request, 'mypage/web/my-club-web.html', context)


class MypageMyClubAPI(APIView):
    def post(self, request):
        member = request.session.get('member')
        order = request.data.get('order', '이름 순')
        page = int(request.data.get('page', 1))
        print(page,order)

        filter = 'name'
        if order == '최신순':
            filter = '-created_date'
        if order == '오래된 순':
            filter ='created_date'
        elif order == '활동 순':
            filter = '-activity_count'


        row_count = 6
        offset = (page - 1) * row_count
        limit = page * row_count

        clubs = Club.enabled_objects.filter(member_id=member['id']) \
            .annotate(name=F('club_name'), profile_path=F('club_profile_path'), activity_count=Count('activity'),
                      alarms=F('clubmember__alarm_status'), join_status=Value(2)) \
            .values('id', 'name', 'profile_path', 'alarms', 'join_status', 'activity_count', 'created_date')

        club_member = ClubMember.objects.filter(member_id=member['id']) \
            .annotate(name=F('club__club_name'), profile_path=F('club__club_profile_path'),
                      activity_count=Count('club__activity'), alarms=F('alarm_status'),
                      join_status=F('status')) \
            .values('club_id', 'name', 'profile_path', 'alarms', 'join_status', 'activity_count', 'created_date')

        combined_data = club_member.union(clubs).order_by(filter)

        club_list = list(combined_data[offset:limit])
        for club in club_list:
            print(club)
        return Response(club_list)


class MypageMyClubAlarmStatusAPI(APIView):
    def post(self, request, club_id):
        member = request.session.get('member')
        club_member_queryset = ClubMember.enabled_objects.filter(member_id=member['id'], club_id=club_id)
        alarm_status = club_member_queryset.values('alarm_status').first()
        if alarm_status['alarm_status']:
            club_member_queryset.update(alarm_status=False)
        else:
            club_member_queryset.update(alarm_status=True)

        return Response('ok')


class MypageMemberView(View):
    def get(self, request, club_id):
        search = request.GET.get('search', '')
        order = request.GET.get('order', '전체 상태')
        page = request.GET.get('page', 1)

        context = {
            'search': search,
            'order': order,
            'page': page,
            'club_id': club_id,
        }
        return render(request, 'mypage/web/management-club-member-web.html', context)

    def post(self, request):
        datas = request.POST
        context = {
            'order': datas.get('order', '최신순'),
            'page': datas.get('page', 1),
            'search': datas.get('search', 1),
        }

        return render(request, 'mypage/web/management-club-member-web.html', context)


class MypageMemberListAPI(APIView):
    def post(self, request, club_id):
        search = request.data.get('search', '')
        order = request.data.get('order', '전체 상태')
        page = int(request.data.get('page', 1))
        row_count = 8
        offset = (page - 1) * row_count
        limit = page * row_count

        condition = Q(club_id=club_id)
        if search:
            condition &= Q(member__member_email__contains=search) | Q(member__member_nickname__contains=search)
        if order == '가입대기':
            condition &= Q(status=-1)
        elif order == '가입중':
            condition &= Q(status=1)

        club_members_queryset = ClubMember.objects.filter(condition).exclude(status=0)

        club_members = list(club_members_queryset.values('id', 'member__member_nickname', 'member__member_email',
                                                         'member__member_gender',
                                                         'member__member_address', 'member__created_date', 'status',
                                                         'member')[offset:limit])
        club_members_count = club_members_queryset.count()

        page_count = 5
        end_page = math.ceil(page / page_count) * page_count
        start_page = end_page - page_count + 1
        real_end = math.ceil(club_members_count / row_count)
        end_page = real_end if end_page > real_end else end_page

        if end_page == 0:
            end_page = 1

        page_info = {
            'totalCount': club_members_count,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }
        for club_member in club_members:
            club_member['member__created_date'] = club_member['member__created_date'].strftime("%Y-%m-%d")
            member_favorite_categories = MemberFavoriteCategory.objects.filter(member=club_member['member']).values(
                'category__category_name')
            club_member['member_favorite_categories'] = list(member_favorite_categories)
            club_member['category_count'] = len(club_member['member_favorite_categories']) - 1

            activities = ActivityMember.objects.filter(member=club_member['member']).values('activity__activity_title')
            club_member['activities'] = list(activities)

        club_members.append(page_info)

        return Response(club_members)


class MypageClubDeleteView(View):
    def get(self, request, club_id):
        club = Club.objects.get(id=club_id)
        activities_count = Activity.enabled_objects.filter(club=club).count()
        club_members_count = ClubMember.objects.filter(club=club, status=-1).count()

        context = {
            'club_id': club_id,
            'club_name': club.club_name,
            'activities_count': activities_count,
            'club_members_count': club_members_count,
        }
        return render(request, 'mypage/web/management-club-delete-web.html', context)

    def post(self, request, club_id):
        club = Club.objects.get(id=club_id)
        club.status = False
        club.save()

        return redirect('/member/mypage-my-club/')


class MypageMemberStatusAPI(APIView):
    def delete(self, request, club_id):
        member = request.data.get('member_id')
        ClubMember.objects.filter(club_id=club_id, id=member).update(status=0)

        return Response('ok')

    def patch(self, request, club_id):
        member = request.data.get('member_id')
        ClubMember.objects.filter(club_id=club_id, id=member).update(status=1)

        return Response('ok')


class MypageNoticeView(View):
    def get(self, request, club_id):
        page = request.GET.get('page', 1)

        context = {
            'page': page,
            'club_id': club_id
        }

        return render(request, 'mypage/web/management-club-notice-web.html', context)


class MypageNoticeAPI(APIView):
    def post(self, request, club_id):
        del_item = request.data.get('del_item', '')
        page = int(request.data.get('page', 1))
        row_count = 5
        offset = (page - 1) * row_count
        limit = page * row_count

        if del_item:
            for item in del_item:
                ClubNotice.objects.filter(id=item, club_id=club_id, status=1).update(status=0)

        club_notices_queryset = ClubNotice.objects.filter(club_id=club_id, status=1)
        club_notices = list(club_notices_queryset.order_by('-id').values('id', 'notice_title', 'notice_content', 'created_date')[offset:limit])
        club_notices_count = club_notices_queryset.count()

        page_count = 5
        end_page = math.ceil(page / page_count) * page_count
        start_page = end_page - page_count + 1
        real_end = math.ceil(club_notices_count / row_count)
        end_page = real_end if end_page > real_end else end_page

        if end_page == 0:
            end_page = 1

        page_info = {
            'totalCount': club_notices_count,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        for club_notice in club_notices:
            club_notice['created_date'] = club_notice['created_date'].strftime("%Y.%m.%d")

        club_notices.append(page_info)

        return Response(club_notices)


class MypageNoticeDeleteAPI(APIView):
    def delete(self, request, club_id):
        del_list = request.data['del_list']
        for del_item in del_list:
            ClubNotice.objects.filter(id=del_item, club_id=club_id, status=1).update(status=0)

        return Response('ok')


class MypageNoticeCreateView(View):
    def get(self, request, club_id):
        context = {'club_id': club_id}
        return render(request, 'mypage/web/management-club-notice-write-web.html', context)

    def post(self, request, club_id):
        title = request.POST.get('title')
        content = request.POST.get('content')
        club = Club.enabled_objects.get(id=club_id)

        data = {
            'club': club,
            'notice_title': title,
            'notice_content': content
        }

        ClubNotice.objects.create(**data)
        # 상세페이지 주소로 이동하게 해야함
        return redirect(f'/member/mypage-notice/{club_id}/')


class MypageNoticeModifyView(View):
    def get(self, request, club_id):
        club_notice_id = request.GET.get('id')
        club_notice = ClubNotice.objects.filter(id=club_notice_id).values('id', 'club_id', 'notice_title',
                                                                          'notice_content').first()

        context = {
            'clubNotice': club_notice,
            'club_id': club_id
        }
        return render(request, 'mypage/web/management-club-notice-modify-web.html', context)

    def post(self, request, club_id):
        club_notice_id = request.GET.get('id')
        title = request.POST.get('title')
        content = request.POST.get('content')

        club_notice = ClubNotice.objects.filter(id=club_notice_id).first()

        if club_notice:
            club_notice.notice_title = title
            club_notice.notice_content = content
            club_notice.save()

            # 상세페이지 주소로 이동하게 해야함
        return redirect(f'/member/mypage-notice/{club_id}')


class MypageSendLetterAPI(APIView):
    def post(self, request):
        letter = request.data.get('letter')

        for receiver in letter['receivers']:
            receiver_start_index = receiver.find('(') + 1
            receiver_end_index = receiver.find(')')
            receiver_email = receiver[receiver_start_index:receiver_end_index]

            receiver = Member.objects.get(member_email=receiver_email)

            sender_start_index = letter['sender'].find('(') + 1
            sender_end_index = letter['sender'].find(')')
            sender_email = letter['sender'][sender_start_index:sender_end_index]

            sender = Member.objects.get(member_email=sender_email)

            data = {
                'sender': sender,
                'receiver': receiver,
                'letter_content': letter['content']
            }

            letter = Letter.objects.create(**data)
            ReceivedLetter.objects.create(letter_id=letter.id)

            SentLetter.objects.create(letter_id=letter.id)
            Alarm.objects.create(target_id=letter.id, alarm_type=4, sender_id=letter.sender_id,
                                 receiver_id=letter.receiver_id)

        return Response('ok')


class MypageSettingView(View):
    def get(self, request, club_id):
        club = Club.objects.get(id=club_id)

        categories = Category.objects.all().distinct()
        club_category_id = Club.objects.filter(id=club_id).first().club_main_category_id
        category_name = Category.objects.filter(id=club_category_id).first().category_name
        club_sub_category_id = ClubCategory.objects.filter(club_id=club_id, status=1).values_list('category_id', flat=True)

        regions = Region.objects.all()
        region_id = Club.objects.filter(id=club_id).first().club_region_id
        region_name = Region.objects.filter(id=region_id).first().region

        context = {
            'club_id': club_id,
            'clubName': club.club_name,
            'clubIntro': club.club_intro,
            'clubInfo': club.club_info,
            'clubProfilePath': club.club_profile_path,
            'clubBannerPath': club.club_banner_path,
            'memberNickname': club.member.member_nickname,
            'memberEmail': club.member.member_email,
            'memberPhone': club.member.member_phone,
            'categories': categories,
            'regions': regions,
            'club_category_id': club_category_id,
            'category_name': category_name,
            'region_id': region_id,
            'region_name': region_name,
            'club_sub_category_id': list(club_sub_category_id),
        }

        return render(request, 'mypage/web/management-club-setting-web.html', context)

    def post(self, request, club_id):
        data = request.POST
        files = request.FILES
        club = Club.objects.get(id=club_id)
        club.club_name = data.get('club-name')
        club.club_intro = data.get('club-intro')
        club.club_info = data.get('club-details')
        club.club_main_category_id = data.get('club-category')
        club.club_region_id = data.get('club-region')

        for key in files:
            if key == 'club-profile':
                club.club_profile_path = files[key]
            elif key == 'background-img':
                club.club_banner_path = files[key]
        club.save()

        # 카테고리 아이디 대조
        # 있으면 수정
        # 업으면 추가

        # 생성한 모임의 서브 카테고리를 저장하기 위해 모임의 id를 객체화
        club.id = club.id

        # 모든 기존 카테고리의 상태를 0으로 설정
        ClubCategory.objects.filter(club_id=club.id).update(status=0)

        # 서브 카테고리를 저장하기 위해 입력 받은 정보를 저장 (문자열로 반환하기 때문에 정수로 변환)
        sub_categories = [int(id) for id in request.POST.getlist('club-sub-category')]

        for sub_category in sub_categories:
            # 해당 카테고리가 이미 존재하는지 확인 (있으면 객체 가져오고, 없으면 생성)
            club_category, created = ClubCategory.objects.get_or_create(club_id=club_id, category_id=sub_category)
            # 상태를 1로 설정
            club_category.status = 1
            # 업데이트 날짜를 현재 시간으로 지정
            club_category.updated_date = timezone.now()
            # 변경사항 저장
            club_category.save()

        member_instance = Member.objects.get(id=club.member_id)
        member_instance.member_nickname = data.get('manager-name')
        member_instance.member_email = data.get('manager-email')
        member_instance.member_phone = data.get('manager-phone')
        member_instance.save()

        return redirect(f'/member/mypage-setting/{club_id}')


class MypageActivityVIEW(View):
    def get(self, request):
        return render(request, 'mypage/web/my-activity-web.html')


class MypageActivityAPIVIEW(APIView):
    @transaction.atomic
    def get(self, request, member_id, page):
        status_like = request.GET.get('status_like')
        print(status_like)
        row_count = 10
        offset = (page - 1) * row_count
        limit = page * row_count

        activity_data = []
        # 클럽장
        activity_data += Activity.objects.filter(activitymember__member_id=member_id, status=1).values('id',
                                                                                                       'created_date',
                                                                                                       'activity_title',
                                                                                                       'thumbnail_path',
                                                                                                       'club__member_id',
                                                                                                       'activity_end',
                                                                                                       'activitymember__status')
        activity_data += Activity.objects.filter(club__member_id=member_id, status=1).values('id', 'created_date',
                                                                                             'activity_title',
                                                                                             'thumbnail_path',
                                                                                             'club__member_id',
                                                                                             'activity_end')

        for activity in activity_data:
            activity_id = activity.get('id')
            activity_like = ActivityLike.enabled_objects.filter(activity_id=activity_id, member_id=member_id)
            activity['status'] = activity_like.exists()

        if status_like == 'like':
            activity_data = ActivityLike.objects.filter(member_id=member_id, status=1).values('id', 'status',
                                                                                              'activity__thumbnail_path',
                                                                                              'activity__activity_end',
                                                                                              'activity__activity_title',
                                                                                              'activity_id')

        return Response(activity_data[offset:limit])


class MypageActivityLikeAPIVIEW(APIView):
    @transaction.atomic
    def patch(self, request, activity_id):
        like, created = ActivityLike.objects.get_or_create(activity_id=activity_id,
                                                           member_id=request.session['member']['id'])
        # 반전시키기
        like.status = 1 - like.status
        like.save()
        return Response('good')


class ActivityManagentView(View):
    def get(self, request):
        activity_id = request.GET.get('activity_id')
        activity = Activity.objects.filter(id=activity_id).first()
        member_id = request.session['member']['id']
        active_count = ActivityMember.objects.filter(activity_id=activity_id, status=1,member_id=member_id).count() + 1
        wait_count = ActivityMember.objects.filter(activity_id=activity_id, status=-1).count()
        total_count = active_count + wait_count
        club = activity.club
        club_id = club.id
        club_notices = list(ClubNotice.objects.filter(status=True, club_id=club.id).order_by('-id')[:4])
        tennplay_notices = list(Notice.objects.filter(notice_type=0).order_by('-id')[:4])
        my_activity = Activity.objects.filter(
            Q(club__member_id=member_id, status = 1) | Q(activitymember__member_id=member_id ,status = 1)).count()


        context = {'activity': activity,
                   'active_count': active_count,
                   'wait_count': wait_count,
                   'total_count': total_count,
                   'club_notices': club_notices,
                   'tennplay_notices': tennplay_notices,
                   'my_activity': my_activity,
                   'club_id':club_id
                   }

        return render(request, 'mypage/web/management-activity-web.html', context)


class ActivityListAPIView(APIView):
    @transaction.atomic
    def get(self, request, member_id, page):
        status_list = request.GET.get('status_list')

        row_count = 5
        offset = (page - 1) * row_count
        limit = page * row_count
        total_count = 0

        total_count += Activity.objects.filter(club__member_id=member_id, status = 1).count()
        total_count += Activity.objects.filter(activitymember__member_id=member_id ,activitymember__status = 1).count()


        total_pages = (total_count + row_count - 1) // row_count

        activity_data = []
        activity_data += Activity.objects.filter(club__member_id=member_id, status = 1).values('id',
                                                                                                                'created_date',
                                                                                                                'activity_title',
                                                                                                                'activity_intro',
                                                                                                                'activity_address_location',
                                                                                                                'category__category_name',
                                                                                                                'club__member_id',
                                                                                                                'activity_end',
                                                                                                                'activity_start').order_by(
            '-activity_start')
        activity_data += Activity.objects.filter(activitymember__member_id=member_id ,activitymember__status = 1).values('id',
                                                                                                                'created_date',
                                                                                                                'activity_title',
                                                                                                                'activity_intro',
                                                                                                                'activity_address_location',
                                                                                                                'category__category_name',
                                                                                                                'club__member_id',
                                                                                                                'activity_end',
                                                                                                                'activity_start').order_by(
            '-activity_start')

        response_data = {
            'total_pages': total_pages,
            'activity_data': activity_data[offset:limit],
        }


        if status_list == 'old/':

            total_count = 0

            total_count += Activity.objects.filter(club__member_id=member_id, status=1).count()
            total_count += Activity.objects.filter(activitymember__member_id=member_id, activitymember__status=1).count()

            total_pages = (total_count + row_count - 1) // row_count

            activity_data = []
            activity_data += Activity.objects.filter(club__member_id=member_id, status=1).values('id',
                                                                                                    'created_date',
                                                                                                    'activity_title',
                                                                                                    'activity_intro',
                                                                                                    'activity_address_location',
                                                                                                    'category__category_name',
                                                                                                    'club__member_id',
                                                                                                    'activity_end',
                                                                                                    'activity_start').order_by(
                'activity_start')
            activity_data += Activity.objects.filter(activitymember__member_id=member_id, activitymember__status=1).values('id',
                                                                                                            'created_date',
                                                                                                            'activity_title',
                                                                                                            'activity_intro',
                                                                                                            'activity_address_location',
                                                                                                            'category__category_name',
                                                                                                            'club__member_id',
                                                                                                            'activity_end',
                                                                                                            'activity_start').order_by(
                'activity_start')

            response_data['activity_data'] = activity_data
            response_data['total_pages'] =  total_pages

        else:
            total_count = 0

            total_count += Activity.objects.filter(club__member_id=member_id, status=1).count()
            total_count += Activity.objects.filter(activitymember__member_id=member_id, activitymember__status=1).count()

            total_pages = (total_count + row_count - 1) // row_count

            activity_data = []
            activity_data += Activity.objects.filter(club__member_id=member_id, status=1).values('id',
                                                                                                    'created_date',
                                                                                                    'activity_title',
                                                                                                    'activity_intro',
                                                                                                    'activity_address_location',
                                                                                                    'category__category_name',
                                                                                                    'club__member_id',
                                                                                                    'activity_end',
                                                                                                    'activity_start').order_by(
                '-activity_start')
            activity_data += Activity.objects.filter(activitymember__member_id=member_id, activitymember__status=1).values('id',
                                                                                                            'created_date',
                                                                                                            'activity_title',
                                                                                                            'activity_intro',
                                                                                                            'activity_address_location',
                                                                                                            'category__category_name',
                                                                                                            'club__member_id',
                                                                                                            'activity_end',
                                                                                                            'activity_start').order_by(
                '-activity_start')


        return Response(response_data)


class ActivityMemberView(View):
    def get(self, request):
        activity_id = request.GET.get('activity_id')
        context = {
            'activity_id': activity_id
        }
        return render(request, 'mypage/web/management-activity-member-web.html', context)


class ActivityMemberListAPIView(APIView):
    @transaction.atomic
    def get(self, request, member_id, page, activity_id):
        member_status = request.GET.get('member_status')
        search = request.GET.get('search')

        row_count = 5 if page != 1 else 4
        offset = (page - 1) * row_count
        limit = page * row_count

        total_count = ActivityMember.objects.filter(
            Q(activity_id=activity_id, status=1) | Q(activity_id=activity_id, status=-1)).count() + 1
        total_pages = (total_count + row_count - 1) // row_count

        if page == 1:
            club_manager = Activity.objects.filter(id=activity_id, club__member_id=member_id).values('id',
                                                                                                     'club__member__member_nickname',
                                                                                                     'club__member__member_email',
                                                                                                     'created_date',
                                                                                                     'club__member__member_gender',
                                                                                                     'club__member__member_birth',
                                                                                                     'club__member__id'
                                                                                                     ).first()

            manager_favorite_categories = list(MemberFavoriteCategory.objects.filter(status=True,
                                                                                     member_id=club_manager.get(
                                                                                         'club__member__id')).values())
            club_manager['categories'] = manager_favorite_categories

        else:
            club_manager = None

        member_list = ActivityMember.objects.filter(
            Q(activity_id=activity_id, status=1) | Q(activity_id=activity_id, status=-1)) \
            .values('id', 'member__member_nickname', 'member__member_email', 'activity__created_date', 'status',
                    'member__member_gender', 'member__member_birth')

        # 새로운 리스트를 만들어 멤버 정보와 카테고리 정보를 함께 저장
        for member_category in member_list:
            member_id = member_category.get('member_id')
            manager_favorite_categories_add = list(
                MemberFavoriteCategory.objects.filter(status=True, member_id=member_id).values())
            member_category['categories'] = manager_favorite_categories_add

        # 이제 updated_member_list를 사용하여 결과를 처리할 수 있음

        response_data = {
            'club_manager': club_manager,
            "member_list": member_list[offset:limit],
            "total_pages": total_pages
        }

        if member_status == 'attending':

            member_list = ActivityMember.objects.filter(activity_id=activity_id, status=1) \
                .values('id', 'member__member_nickname', 'member__member_email', 'activity__created_date', 'status',
                        'member__member_gender', 'member__member_birth')

            # 새로운 리스트를 만들어 멤버 정보와 카테고리 정보를 함께 저장
            for member_category in member_list:
                member_id = member_category.get('member_id')
                manager_favorite_categories_add = list(
                    MemberFavoriteCategory.objects.filter(status=True, member_id=member_id).values())
                member_category['categories'] = manager_favorite_categories_add

            total_count = ActivityMember.objects.filter(activity_id=activity_id, status=1).count() + 1
            total_pages = (total_count + row_count - 1) // row_count
            response_data['member_list'] = member_list
            response_data['total_pages'] = total_pages

            if search:
                member_list = ActivityMember.objects.filter(
                    Q(activity_id=activity_id, status=1, member__member_nickname__icontains=search) | Q(
                        activity_id=activity_id, status=1, member__member_email__icontains=search)) \
                    .values('id', 'member__member_nickname', 'member__member_email', 'activity__created_date', 'status',
                            'member__member_gender', 'member__member_birth')

                # 새로운 리스트를 만들어 멤버 정보와 카테고리 정보를 함께 저장
                for member_category in member_list:
                    member_id = member_category.get('member_id')
                    manager_favorite_categories_add = list(
                        MemberFavoriteCategory.objects.filter(status=True, member_id=member_id).values())
                    member_category['categories'] = manager_favorite_categories_add

                total_count = ActivityMember.objects.filter(activity_id=activity_id, status=1).count() + 1
                total_pages = (total_count + row_count - 1) // row_count
                response_data['member_list'] = member_list
                response_data['total_pages'] = total_pages


        elif member_status == 'waiting':

            member_list = ActivityMember.objects.filter(activity_id=activity_id, status=-1) \
                .values('id', 'member__member_nickname', 'member__member_email', 'activity__created_date', 'status',
                        'member__member_gender', 'member__member_birth')

            # 새로운 리스트를 만들어 멤버 정보와 카테고리 정보를 함께 저장
            for member_category in member_list:
                member_id = member_category.get('member_id')
                manager_favorite_categories_add = list(
                    MemberFavoriteCategory.objects.filter(status=True, member_id=member_id).values())
                member_category['categories'] = manager_favorite_categories_add

            total_count = ActivityMember.objects.filter(activity_id=activity_id, status=-1).count() + 1
            total_pages = (total_count + row_count - 1) // row_count
            response_data['member_list'] = member_list
            response_data['total_pages'] = total_pages

            if search:
                member_list = ActivityMember.objects.filter(
                    Q(activity_id=activity_id, status=-1, member__member_nickname__icontains=search) | Q(
                        activity_id=activity_id, status=-1, member__member_email__icontains=search)) \
                    .values('id', 'member__member_nickname', 'member__member_email', 'activity__created_date', 'status',
                            'member__member_gender', 'member__member_birth')

                # 새로운 리스트를 만들어 멤버 정보와 카테고리 정보를 함께 저장
                for member_category in member_list:
                    member_id = member_category.get('member_id')
                    manager_favorite_categories_add = list(
                        MemberFavoriteCategory.objects.filter(status=True, member_id=member_id).values())
                    member_category['categories'] = manager_favorite_categories_add

                total_count = ActivityMember.objects.filter(activity_id=activity_id, status=-1).count() + 1
                total_pages = (total_count + row_count - 1) // row_count
                response_data['member_list'] = member_list
                response_data['total_pages'] = total_pages

        else:
            member_list = ActivityMember.objects.filter(
                Q(activity_id=activity_id, status=1) | Q(activity_id=activity_id, status=-1)) \
                .values('id', 'member__member_nickname', 'member__member_email', 'activity__created_date', 'status',
                        'member__member_gender', 'member__member_birth')

            # 새로운 리스트를 만들어 멤버 정보와 카테고리 정보를 함께 저장
            for member_category in member_list:
                member_id = member_category.get('member_id')
                manager_favorite_categories_add = list(
                    MemberFavoriteCategory.objects.filter(status=True, member_id=member_id).values())
                member_category['categories'] = manager_favorite_categories_add

            total_count = ActivityMember.objects.filter(
                Q(activity_id=activity_id, status=1) | Q(activity_id=activity_id, status=-1)).count() + 1
            total_pages = (total_count + row_count - 1) // row_count
            response_data['member_list'] = member_list
            response_data['total_pages'] = total_pages

            if search:
                member_list = ActivityMember.objects.filter(
                    Q(activity_id=activity_id, status=1, member__member_nickname__icontains=search) | Q(
                        activity_id=activity_id, status=-1, member__member_nickname__icontains=search) | Q(
                        activity_id=activity_id, status=1, member__member_email__icontains=search) | Q(
                        activity_id=activity_id, status=-1, member__member_email__icontains=search)) \
                    .values('id', 'member__member_nickname', 'member__member_email', 'activity__created_date', 'status',
                            'member__member_gender', 'member__member_birth')

                # 새로운 리스트를 만들어 멤버 정보와 카테고리 정보를 함께 저장
                for member_category in member_list:
                    member_id = member_category.get('member_id')
                    manager_favorite_categories_add = list(
                        MemberFavoriteCategory.objects.filter(status=True, member_id=member_id).values())
                    member_category['categories'] = manager_favorite_categories_add

                total_count = ActivityMember.objects.filter(
                    Q(activity_id=activity_id, status=1) | Q(activity_id=activity_id, status=-1)).count() + 1
                total_pages = (total_count + row_count - 1) // row_count
                response_data['member_list'] = member_list
                response_data['total_pages'] = total_pages

        return Response(response_data)


class ActivityMemberUpdateAPIView(APIView):
    @transaction.atomic
    def patch(self, request, activity_member_id, activity_id):
        ActivityMember.objects.filter(id=activity_member_id).update(status=1,updated_date=timezone.now())
        updated_member = ActivityMember.objects.get(id=activity_member_id)
        receiver_id = updated_member.member_id
        member_id = request.session['member']['id']
        Alarm.objects.create(target_id=activity_id,receiver_id=receiver_id,sender_id=member_id, alarm_type=12)
        print('asd')
        return Response('good')

    @transaction.atomic
    def delete(self, request, activity_member_id, activity_id):
        ActivityMember.objects.filter(id=activity_member_id).update(status=0,updated_date=timezone.now())
        updated_member = ActivityMember.objects.get(id=activity_member_id)
        receiver_id = updated_member.member_id
        member_id = request.session['member']['id']
        Alarm.objects.create(target_id=activity_id, receiver_id=receiver_id,
                             sender_id=member_id, alarm_type=13)

        return Response('good')


class ActivityMemberWriteAPI(APIView):
    @transaction.atomic
    def post(self, request):
        data = request.data
        receiver_email = data['receiver_email']
        receiver_instance = Member.objects.filter(member_email=receiver_email).first()
        receiver_id = receiver_instance.id
        # Letter 모델 생성
        letter = Letter.objects.create(
            receiver_id=receiver_id,
            letter_content=data['letter_content'],
            sender_id=request.session['member']['id']
        )

        # ReceivedLetter 모델 생성
        ReceivedLetter.objects.create(letter_id=letter.id)

        # SentLetter 모델 생성
        SentLetter.objects.create(letter_id=letter.id)
        Alarm.objects.create(target_id=letter.id, alarm_type=4, sender_id=letter.sender_id,
                             receiver_id=letter.receiver_id)

        return Response("good")


class ActivityEditView(View):
    def get(self, request):
        activity_id = request.GET.get('activity_id')
        activity = Activity.objects.filter(id=activity_id).first()
        member_id = request.session['member']['id']
        member = Member.objects.filter(id=member_id).first()
        categories = Category.objects.filter(status=True)

        context = {'activity': activity,
                   'member': member,
                   'categories': categories,
                   }

        return render(request, 'mypage/web/management-activity-edit-web.html', context)

    @transaction.atomic
    def post(self, request):
        data = request.POST
        activity = Activity.objects.get(id=data.get('activity-id'))
        print('------')
        print(activity.activity_title)
        print('------')
        a = data.get('activity-title')
        recruit_start = make_datetime(data.get('recruit-start-date').replace(" ", ""), data.get('recruit-start-time'))
        recruit_end = make_datetime(data.get('recruit-end-date').replace(" ", ""), data.get('recruit-end-time'))
        category = Category.objects.get(id=data.get('category'))
        thumbnail = request.FILES.get('thumbnail-path')
        banner = request.FILES.get('banner-path')
        activity_start = make_datetime(data.get('activity-start-date').replace(" ", ""),
                                       data.get('activity-start-time'))
        activity_end = make_datetime(data.get('activity-end-date').replace(" ", ""), data.get('activity-end-time'))

        activity.activity_title = a
        activity.activity_content = data.get('activity-content')
        activity.recruit_start = recruit_start
        activity.recruit_end = recruit_end
        activity.category = category
        activity.activity_intro = data.get('activity-intro')
        if thumbnail is not None:
            activity.thumbnail_path = thumbnail
        if banner is not None:
            activity.banner_path = banner
        activity.activity_address_location = data.get('activity-address-location')
        activity.activity_address_detail = data.get('activity-address-detail')

        activity.activity_start = activity_start
        activity.activity_end = activity_end

        activity.save()

        return redirect(f'/member/activity-edit?activity_id={activity.id}')


class MypageWishlistWebView(View):
    def get(self, request):
        member = request.session.get('member')
        # latest_letter = Letter.objects.latest('id')  # 또는 필요한 기준으로 필터링
        return render(request, 'mypage/web/my-wishlist-web.html', {"member": member})


class MypageWishlistAPIView(APIView):
    @transaction.atomic
    def get(self, request, member_id, page):
        status_wishlist = request.GET.get('status_wishlist')
        row_count = 10
        offset = (page - 1) * row_count
        limit = page * row_count

        # 전체 데이터 수를 세는 쿼리
        total_count = Wishlist.objects.filter(Q(member_id=member_id, status=1)).count()

        # 전체 페이지 수 계산
        total_pages = (total_count + row_count - 1) // row_count

        # 페이지 범위에 해당하는 데이터 조회
        # 전체
        wishlists = Wishlist.objects.filter(Q(member_id=member_id), status=1).annotate(
            like_total=Count('wishlistlike__id', filter=Q(wishlistlike__status=1)),
            reply_total=Count('wishlistreply__id', filter=Q(wishlistreply__status=1)),
            category_name=F("category__category_name")) \
                        .values('wishlist_content', 'created_date', 'is_private', 'category_name', 'category',
                                'reply_total', 'like_total',
                                'id')[offset:limit]
        # 공개
        open = Wishlist.objects.filter(member_id=member_id, is_private=1, status=1).annotate(
            like_total=Count('wishlistlike__id', filter=Q(wishlistlike__status=1)),
            reply_total=Count('wishlistreply__id', filter=Q(wishlistreply__status=1)),
            category_name=F("category__category_name")) \
                   .values('wishlist_content', 'created_date', 'is_private', 'category_name', 'category', 'reply_total',
                           'like_total',
                           'id')[offset:limit]
        # 비공개
        secret = Wishlist.objects.filter(member_id=member_id, is_private=0, status=1).annotate(
            like_total=Count('wishlistlike__id', filter=Q(wishlistlike__status=1)),
            reply_total=Count('wishlistreply__id', filter=Q(wishlistreply__status=1)),
            category_name=F("category__category_name")) \
                     .values('wishlist_content', 'created_date', 'is_private', 'category_name', 'category',
                             'reply_total', 'like_total',
                             'id')[offset:limit]

        # 응답에 전체 페이지 수와 데이터 추가
        response_data = {
            'total_pages': total_pages,
            'wishlists': wishlists,

        }
        if status_wishlist == 'open':
            response_data['wishlists'] = Wishlist.objects.filter(member_id=member_id, is_private=1, status=1).annotate(
                like_total=Count('wishlistlike__id', filter=Q(wishlistlike__status=1),
                                 category=F('category__category_name')),
                reply_total=Count('wishlistreply__id', filter=Q(wishlistreply__status=1)),
                category_name=F("category__category_name")) \
                                             .values('wishlist_content', 'created_date', 'is_private', 'category_name',
                                                     'category',
                                                     'reply_total', 'like_total',
                                                     'id')[offset:limit]

            response_data['total_pages'] = (Wishlist.objects.filter(member_id=member_id, is_private=1,
                                                                    status=1).count() + row_count - 1) // row_count
        elif status_wishlist == 'secret':
            response_data['wishlists'] = Wishlist.objects.filter(member_id=member_id, is_private=0, status=1).annotate(
                like_total=Count('wishlistlike__id', filter=Q(wishlistlike__status=1)),
                reply_total=Count('wishlistreply__id', filter=Q(wishlistreply__status=1)),
                category_name=F("category__category_name")) \
                                             .values('wishlist_content', 'created_date', 'is_private', 'category_name',
                                                     'category',
                                                     'reply_total', 'like_total',
                                                     'id')[offset:limit]
            response_data['total_pages'] = (Wishlist.objects.filter(member_id=member_id, is_private=0,
                                                                    status=1).count() + row_count - 1) // row_count

        else:
            response_data['wishlists'] = Wishlist.objects.filter(Q(member_id=member_id, status=1)).annotate(
                like_total=Count('wishlistlike__id', filter=Q(wishlistlike__status=1)),
                reply_total=Count('wishlistreply__id', filter=Q(wishlistreply__status=1)),
                category_name=F("category__category_name")) \
                                             .values('wishlist_content', 'created_date', 'is_private', 'category_name',
                                                     'category',
                                                     'reply_total', 'like_total', 'id')[offset:limit]

            response_data['total_pages'] = (Wishlist.objects.filter(member_id=member_id,
                                                                    status=1).count() + row_count - 1) // row_count

        return Response(response_data)


class MypageWishlistDeleteAPIView(APIView):
    @transaction.atomic
    def delete(self, request, wishlist_id):
        wishlist = Wishlist.objects.get(id=wishlist_id)
        wishlist.status = 0
        # Wishlist.objects.filter(wishlist_id=wishlist_id).update(status=0)
        wishlist.save(update_fields=['status'])

        return Response('success')