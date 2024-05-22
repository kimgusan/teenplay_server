import os
from pathlib import Path

from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
import joblib
import numpy as np

from activity.models import Activity, ActivityLike, ActivityMember
from activity.serializers import ActivitySerializer
from club.models import Club, ClubMember
from club.serializers import ClubSerializer
from member.models import Member, MemberFavoriteCategory


# Create your views here.

class RecommendedActivityAPIView(APIView):
    def get(self, request):
        member = request.session.get('member')
        if not member:
            model = joblib.load(os.path.join(Path(__file__).resolve().parent, 'ai/activity_recommender.pkl'))
            probabilities = model.predict_proba(['여행 바다 산 여름 시원한'])
            showing_categories = probabilities.argsort()[0][::-1] + 1
            recommended_activities = list(Activity.enabled_objects.filter(category_id=showing_categories[0]).order_by('-id')[:6])
            recommended_activities += list(Activity.enabled_objects.filter(category_id=showing_categories[1]).order_by('-id')[:2])
            result_activities = []
            for activity in recommended_activities:
                serial_activity = ActivitySerializer(activity).data
                serial_activity['is_like'] = False
                serial_activity['like_count'] = ActivityLike.enabled_objects.filter(activity=activity).count()
                serial_activity['member_count'] = ActivityMember.enabled_objects.filter(activity=activity).count()
                result_activities.append(serial_activity)

            data = {
                'activities': result_activities
            }

            return Response(data)

        member = Member.enabled_objects.get(id=member.get('id'))
        model = joblib.load(member.member_recommended_activity_model)
        member_favorite_categories = MemberFavoriteCategory.objects.filter(member=member, status=True)
        member_favorite_categories = " ".join([category.category.category_name for category in member_favorite_categories])
        member_keywords = " ".join([member.member_keyword1, member.member_keyword2, member.member_keyword3])

        result = member_favorite_categories + " " + member_keywords
        result += " " + member.member_address
        probabilities = model.predict_proba([result])
        showing_categories = probabilities.argsort()[0][::-1] + 1
        recommended_activities = list(
            Activity.enabled_objects.filter(category_id=showing_categories[0]).order_by('-id')[:6])
        recommended_activities += list(
            Activity.enabled_objects.filter(category_id=showing_categories[1]).order_by('-id')[:2])
        result_activities = []
        for activity in recommended_activities:
            serial_activity = ActivitySerializer(activity).data
            serial_activity['is_like'] = ActivityLike.enabled_objects.filter(activity=activity, member=member).exists()
            serial_activity['like_count'] = ActivityLike.enabled_objects.filter(activity=activity).count()
            serial_activity['member_count'] = ActivityMember.enabled_objects.filter(activity=activity).count()
            result_activities.append(serial_activity)

        data = {
            'activities': result_activities
        }

        return Response(data)


class RecommendedAClubAPIView(APIView):
    def get(self, request):
        # 멤버 정보를 섹션에서 받아 와서 멤버 객체로 생성
        member = request.session.get('member')

        if not member:
            # 사전 훈련 모델을 가져와서 객체 생성
            model = joblib.load(os.path.join(Path(__file__).resolve().parent, 'ai/club.pkl'))
            # 훈련 (훈련 키워드는 특정 시즌(ex_여름 시즌, 겨울 시즌 등)에 맞춰 팀 회의 진행 후 지정)
            # 카테고리 별 확률 순서 객체 생성
            probabilities = model.predict_proba(['여행 바다 산 여름 시원한'])
            # 훈련 결과를 인덱스로 변환하고 내림차순으로 정렬된 카테고리 객체 생성 (argsort: 인덱스로 반환)
            # 카테고리는 0부터 시작이 아니고 1부터 시작이기 때문에 1을 더하여 카테고리 순서를 맞추기
            showing_categories = probabilities.argsort()[0][::-1] + 1
            # 화면에 보여질 카테고리 별 비율(개수) 지정
            # 0번째(가장 비율이 높은 카테고리의 모임)을 6개 보여주기, 1번째(두번째로 비율이 높은 카테고리의 모임)을 2개 보여주기 (화면에 8개 추천)
            recommended_clubs = list(Club.enabled_objects.filter(club_main_category_id=showing_categories[0]).order_by('-id')[:6])
            recommended_clubs += list(Club.enabled_objects.filter(club_main_category_id=showing_categories[1]).order_by('-id')[:2])
            # 모임의 정보를 담을 객체 선언
            result_clubs = []
            # 모임 정보 객체를 반복하여 저장:
            for club in recommended_clubs:
                serial_club = ClubSerializer(club).data
                serial_club['club_profile_path'] = Club.enabled_objects.filter(id=club.id).values('club_profile_path').first()
                serial_club['club_name'] = Club.enabled_objects.filter(id=club.id).values('club_name').first()
                serial_club['activity_count'] = Activity.enabled_objects.filter(club_id=club.id).count()
                serial_club['login'] = 0
                result_clubs.append(serial_club)

            data = {
                'clubs': result_clubs
            }

            return Response(data)

        # 회원의 정보를 가져오기
        member = Member.enabled_objects.get(id=member.get('id'))
        # 해당하는 member의 개인용 모델 저장
        model = joblib.load(member.member_recommended_club_model)
        # 해당 멤버의 지역을 가져오기
        member_address = member.member_address
        # 멤버의 선호 카테고리 객체 생성
        member_favorite_categories = MemberFavoriteCategory.objects.filter(member=member, status=True)
        # 멤버의 선호 카테고리 연결
        member_favorite_categories = " ".join([category.category.category_name for category in member_favorite_categories])
        # 멤버의 키워드 가져와서 연결
        member_keywords = " ".join([member.member_keyword1, member.member_keyword2, member.member_keyword3])

        # 멤버의 지역과 카테고리를 연결
        if member_address is None or member_address.strip() == '':
            result = " " + " " + member_favorite_categories + " " + member_keywords
        else:
            result = member_address + " " + member_favorite_categories + " " + member_keywords

        # 훈련 후 카테고리의 확률 순서를 나타내는 객체 생성
        probabilities = model.predict_proba([result])
        # 훈련 결과를 인덱스로 변환하고 내림차순으로 정렬된 카테고리 객체 생성
        showing_categories = probabilities.argsort()[0][::-1] + 1
        # 화면에 보여질 카테고리 별 비율(개수) 지정
        recommended_clubs = list(
            Club.enabled_objects.filter(club_main_category_id=showing_categories[0]).order_by('-id')[:6])
        recommended_clubs += list(
            Club.enabled_objects.filter(club_main_category_id=showing_categories[1]).order_by('-id')[:2])
        # 모임의 정보를 담을 객체 선언
        result_clubs = []
        # 모임 정보 객체를 반복하여 저장:
        for club in recommended_clubs:
            serial_club = ClubSerializer(club).data
            serial_club['club_profile_path'] = Club.enabled_objects.filter(id=club.id).values('club_profile_path').first()
            serial_club['club_name'] = Club.enabled_objects.filter(id=club.id).values('club_name').first()
            serial_club['activity_count'] = Activity.enabled_objects.filter(club_id=club.id).count()
            serial_club['login'] = 1

            # 멤버가 각 모임의 모임장인지 여부
            serial_club['is_manager'] = club.member_id == member.id

            # 멤버가 각 모임의 구성원인지 여부
            is_member = ClubMember.objects.filter(club_id=club.id, member_id=member.id)

            # 만약 해당 멤버가 모임의 구성원이 맞다면:
            if is_member.exists():
                # first()를 통해 해당 객체 가져온 뒤 저장
                is_member = is_member.first()
                # is_member에 status 상태 저장
                # -1: 가입 대기, 1: 구성원, 0: 탈퇴 또는 구성원이 아닌 상태
                is_member = is_member.status
            # 모임의 구성원이 아닌 경우:
            else:
                is_member = 0

            serial_club['is_member'] = is_member

            result_clubs.append(serial_club)

        data = {
            'clubs': result_clubs,
        }

        return Response(data)




# 사전 학습 모델과 sklearn 버전이 맞지 않는 오류 발생
# jupyter notebook에서 pip install --upgrade scikit-learn==1.4.2
# 또는 django 버전 수정 pip install scikit-learn==1.2.2