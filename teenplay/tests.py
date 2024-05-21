# from datetime import datetime
# from django.db.models.functions import Rank, RowNumber
# from django.db.models import Count, F, Window, Q, Exists
# from django.test import TestCase
# from django.utils import timezone
# from activity.models import Activity
# from django.shortcuts import render, redirect
# from django.views import View
# from rest_framework.views import APIView
# import os.path
# from pathlib import Path
# import joblib
# from rest_framework.response import Response
# import numpy as np
# from sklearn.preprocessing import Binarizer
# from django.db import transaction
# from django.db.models.functions import RowNumber
# from random import randint, choice
# from teenplay.models import TeenPlay, TeenPlayLike
# from member.models import MemberFavoriteCategory, Member
# from wishlist.models import Wishlist
# import sklearn
# from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from django.test import TestCase
from django.db import connection

from django.db import transaction, connection
from django.db.models import Count, Q, F, Exists, Window
from django.db.models.functions import RowNumber
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views import View
from rest_framework.response import Response
from rest_framework.views import APIView
from random import randint, choice

from club.models import Club
from member.models import Member, MemberFavoriteCategory
from teenplay.models import TeenPlay, TeenPlayLike
from teenplay.serializers import TeenplaySerializer
from wishlist.models import Wishlist


# # 매인 페이지 틴플레이 재생을 위한 View
# class TeenplayMainListWebView(View):
#     # 틴플레이 화면으로 이동하기 위한 get 함수
#     def get(self, request):
#         # 로그인된 사용자가 있는 경우 해당 로그인 정보를 가져오기 위한 함수 사용
#         member, id = self.session_member_info(request)
#         # 최초 틴플레이 정보를 뿌리기 위한 랜덤 함수
#         context = self.main_random_list(3)
#         # 틴플레이 페이지 이동을 위한 render 함수 사용, context 데이터와 member 데이터를 화면에 가져가기 위해 context 에 딕셔너리 형태로 사용
#         return render(request, 'teenplay/web/teenplay-play-web.html', {'context': context, 'member': member})
#
#     # 틴플레이 좋아요 클릭 시 사용자가 session에 있는지 확인하기 위한 메소드
#     def session_member_info(self, request):
#         # member 객체가 session에 담겨있다면 member_id, member 객체를 member 변수에 담는다.
#         if 'member' in request.session and 'id' in request.session['member']:
#             id = request.session['member']['id']
#             member = request.session['member']
#         else:
#             # member 객체가 session에 있으면
#             if 'member' in request.session:
#                 # member session에 id 가 있으면 id 를 가져오고 없을 경우 None으로 대입한다
#                 id = request.session['member'].get('id', None)
#             else:
#                 # 둘다 없을 경우 None 으로 대입한다.
#                 id = None
#                 member = None
#         # 함수의 결과를 다른곳에서 사용하기 위해 member와 id를 리턴한다.
#         return member, id
#
#     # 메인 화면에서 랜덤한 리스트를 5개 뿌리기 위해서 사용하는 함수
#     def main_random_list(self, id):
#         # 위시리스트, 맴버 관심 분야, 좋아요 최고 카테고리 분야 훈련된 모델을 통해 top3 호출
#         teenplay_category_top3 = self.get_user_features(id)
#         teenplay_count = TeenPlay.enable_objects.filter(club__club_main_category_id__in=teenplay_category_top3).count()
#
#         # 전체 틴플레이 리스트에서 컬럼이 id 에 대한 부분을 쿼리셋 객체에서 list 로 형변환
#         teenplay_id_value = list(
#             TeenPlay.enable_objects.filter(club__club_main_category_id__in=teenplay_category_top3).values('id'))
#         # 리스트로 형변환 된 내용중 id에 대한 값을 리스트에 담기 위해 빈 리스트 생성
#         teenplay_id_list = []
#         # 리스트 내부에 딕셔너리 중 value 값만 추출하여 빈 리스트에 추가
#         for teenplay_id_dict in teenplay_id_value:
#             teenplay_id_list.append(teenplay_id_dict['id'])
#
#         # 조회한 결과는 딕셔너리 형태로 확인되기 때문에 해당 내용을 리스트에 담기 위해서 빈 리스트 생성
#         teenplay_list = []
#         # 5번 반복을 하여 내용을 가져오기 위해 range 함수 생성
#         for number in range(5):
#             # 좋아요 수를 담을 빈 딕셔너리 생
#             like_count = {}
#
#             # 고정된 리스트에서 랜덤한 id 를 1개씩 추출하기 위해 random 내장 라이브러리의 choice 함수 사용
#             radiant_teenplay = choice(teenplay_id_list)
#             # 틴플레이 객체에서 집계함수, 해당 집계함수의 조건(filter) 와 장고 ORM의 특징을 이용해 다른 테이블의 컬럼 결과를 가져와서 teenplay 변수 대입
#             teenplay = TeenPlay.objects.filter(id=radiant_teenplay, status=1).annotate(
#                 likes=Count('teenplaylike__status', filter=Q(teenplaylike__status=1))).values('id', 'video_path',
#                                                                                               'club__id',
#                                                                                               'club__club_name',
#                                                                                               'club__club_intro',
#                                                                                               'club__club_profile_path',
#                                                                                               'club_id', 'likes',
#                                                                                               'teenplay_title')
#
#             # 사용자가 해당 틴플레이를 좋아요를 눌렀는지 확인하기 위해서 존재 유무 확인, teenplay_id는 위에서 생성한 랜덤한 id를 동시에 대입
#             member_like = TeenPlayLike.objects.filter(member_id=id, teenplay_id=radiant_teenplay, status=1).exists()
#             # 빈 딕셔너리에 사용자가 좋야요 한 결과를 key value 값으로 대입
#             like_count['like_check'] = member_like
#             # teenplay_like 리스트에 언패킹의 특성을 이용해서 각각의 딕셔너리를 하나씩 추출해서 새로운 딕셔너리로 합치기
#             teenplay_like = {**like_count, **teenplay[0]}
#             # 리스트에 새로 만든 딕셔너리 추가
#             teenplay_list.append(teenplay_like)
#
#         context = teenplay_list
#         # 반복문을 통해 생성된 리스트를 리턴값으로 전달
#         return context
#
#     def get_user_features(self, id):
#         # 1. 사용자의 카테고리 id
#         member_like_random_category = MemberFavoriteCategory.objects.filter(member_id=id).order_by('?').values(
#             'category_id').first()
#         # 2. 위시리스트의 카테고리 id
#         wishlist_category = Wishlist.objects.filter(member_id=id, status=1).values('category_id').first()
#         # 3. 최고 좋아요 클럽의 카테고리 id
#         teenplay_like_most_category = (TeenPlayLike.objects.filter(member_id=id, status=1)
#                                        .annotate(category_id=F('teenplay__club__club_main_category_id'))
#                                        .values('category_id')
#                                        .annotate(category_count=Count('category_id'))
#                                        .order_by('-category_count')
#                                        .first())
#
#         if member_like_random_category is None:
#             member_like_random_category = {'category_id': 13}
#         if wishlist_category is None:
#             wishlist_category = {'category_id': member_like_random_category['category_id']}
#         if teenplay_like_most_category is None:
#             teenplay_like_most_category = {'category_id': member_like_random_category['category_id']}
#
#         context = [teenplay_like_most_category['category_id'], member_like_random_category['category_id'],
#                    wishlist_category['category_id']]
#
#         return context
#
# # 틴플레이 리스트를 추가하여 받아올 때 사용하는 API (아래로 스크롤)
# class TeenplayMainListAPIView(APIView):
#     # 해당 url 로 호출을 받으면
#     def get(self, request, slideNumber):
#         teenplay_count = TeenPlay.enable_objects.all().count()
#         print(request)
#         print('들어가있어라!')
#         # 멤버가 현재 세션에 등록되어있는지 확인하는 조건문
#         if 'member' in request.session and 'id' in request.session['member']:
#             id = request.session['member']['id']
#         else:
#             if 'member' in request.session:
#                 id = request.session['member'].get('id', None)
#             else:
#                 id = None
#
#         teenplay_category_top3 = self.get_user_features(id)
#         print(teenplay_category_top3)
#         print('들어가 있어라!!!!')
#
#         # 쿼리셋 객체를 리스트로 변경
#         teenplay_id_value = list(TeenPlay.enable_objects.values('id'))
#         teenplay_id_list = []
#         # 변경된 리스트 내부에 있는 딕셔너리 객체를 id 만 추출하여 빈 리스트에 추가
#         for teenplay_id_dict in teenplay_id_value:
#             teenplay_id_list.append(teenplay_id_dict['id'])
#
#
#         teenplay_list = []
#         # 스크롤 진행 시 random 한 id 를 몇개 가져올 지 정하는 반복문
#         random_count = {'random_count': 30}
#         for number in range(random_count['random_count']):
#             like_count = {}
#
#             # 고정된 리스트에서 랜덤한 id 를 1개씩 추출하기 위해 random 내장 라이브러리의 choice 함수 사용
#             radiant_teenplay = choice(teenplay_id_list)
#             # 틴플레이 객체에서 집계함수, 해당 집계함수의 조건(filter) 와 장고 ORM의 특징을 이용해 다른 테이블의 컬럼 결과를 가져와서 teenplay 변수 대입
#             teenplay = TeenPlay.objects.filter(id=radiant_teenplay, status=1).annotate(
#                 likes=Count('teenplaylike__status', filter=Q(teenplaylike__status=1))).values('id', 'video_path','club__club_name','club__club_intro','club__club_profile_path','club_id', 'likes', 'teenplay_title')
#             # 사용자가 해당 틴플레이를 좋아요를 눌렀는지 확인하기 위해서 존재 유무 확인, teenplay_id는 위에서 생성한 랜덤한 id를 동시에 대입
#             member_like = TeenPlayLike.objects.filter(member_id=id, teenplay_id=radiant_teenplay, status=1).exists()
#             # 빈 딕셔너리에 사용자가 좋야요 한 결과를 key value 값으로 대입
#             like_count['like_check'] = member_like
#             # teenplay_like 리스트에 언패킹의 특성을 이용해서 각각의 딕셔너리를 하나씩 추출해서 새로운 딕셔너리로 합치기
#             teenplay_like = {**like_count, **teenplay[0],**random_count}
#             # 리스트에 새로 만든 딕셔너리 추가
#             teenplay_list.append(teenplay_like)
#
#         context = teenplay_list
#         # 반복문을 통해 생성된 리스트를 리턴 결과로 전달
#         return Response(context)
#
#     def get_user_features(self, id):
#         # 1. 사용자의 카테고리 id
#         member_like_random_category = MemberFavoriteCategory.objects.filter(member_id=id).order_by('?').values(
#             'category_id').first()
#         # 2. 위시리스트의 카테고리 id
#         wishlist_category = Wishlist.objects.filter(member_id=id, status=1).values('category_id').first()
#         # 3. 최고 좋아요 클럽의 카테고리 id
#         teenplay_like_most_category = (TeenPlayLike.objects.filter(member_id=id, status=1)
#                                        .annotate(category_id=F('teenplay__club__club_main_category_id'))
#                                        .values('category_id')
#                                        .annotate(category_count=Count('category_id'))
#                                        .order_by('-category_count')
#                                        .first())
#
#         if member_like_random_category is None:
#             member_like_random_category = {'category_id': 13}
#         if wishlist_category is None:
#             wishlist_category = {'category_id': member_like_random_category['category_id']}
#         if teenplay_like_most_category is None:
#             teenplay_like_most_category = {'category_id': member_like_random_category['category_id']}
#
#         context = [teenplay_like_most_category['category_id'], member_like_random_category['category_id'],
#                    wishlist_category['category_id']]
#
#         return context
#
#
#
#
#
# # 틴플레이 좋아요를 클릭했을 때 동작하는 APIView
# class TeenPlayLikeAPIView(APIView):
#     # 조회가 아닌 Create, Update 를 사용할 떄 오류로 인한 데이터 변경을 막기 위한 transaction.atomic 데코레이터 사용
#     @transaction.atomic
#     # 화면에서 조회해 온 데이터를 get 방식을 통해 데이터를 받아옴
#     def get(self, request, teenplayId, memberSessionId, displayStyle):
#         # member_id, teenplay_id 를 딕셔너리 형태로 생성
#         data = {
#             'member_id': memberSessionId,
#             'teenplay_id': teenplayId
#         }
#
#         # 좋아요 클릭 시 없으면 생성되고 있으면 update 를 하기 위한 get_or_create 함수 사용
#         likeData, checked = TeenPlayLike.objects.get_or_create(**data)
#         # 만약 해당 데이터가 있다면 해당 틴플레이 id 의 total count 를 변수에 대입
#         if checked:
#             totalLikeCount = TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId).count()
#         # 해당 좋아요를 새로 생성했을 때 데이터가 있다면 아래 조건 사용
#         else:
#             # 데이터가 존재하며 현재 좋아요가 클릭되어 있지 않으면
#             if displayStyle != 'none':
#                 # 좋아요 status 를 1로 업데이트 하고 총 카운트를 total count 에 대입한다
#                 TeenPlayLike.objects.filter(status=0, teenplay_id=teenplayId, member_id=memberSessionId).update(status=1, updated_date=timezone.now())
#                 totalLikeCount = TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId).count()
#             else:
#                 # 좋아요 status 를 0으로 업데이트 하고 총 카운트를 total count 에 대입한다
#                 TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId, member_id=memberSessionId).update(status=0, updated_date=timezone.now())
#                 totalLikeCount = TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId).count()
#
#         # 좋아요를 클릭한 결과와 총 count 와 해당 member_id와 display style을 context 에 딕셔너리 형태로 대입하여 응답을 반환한다.
#         context = {
#             'teenplay_id': teenplayId, # teenplay_id
#             'member_id': memberSessionId,
#             'display_style': displayStyle,
#             'totalLikeCount': totalLikeCount
#         }
#
#         return Response(context)
#
# # 클럽에서 틴플레이를 선택했을 떄 사용되는 함수
# def get_teenplay_data(clubId, page, teenplayClickId, member_id):
#     videos = TeenPlay.enable_objects.filter(club_id=clubId).annotate(row_number=Window(expression=RowNumber(), order_by=F('id').desc()))
#     # 각 인스턴스에 대해 순차 번호(행 번호)를 부여
#     # 우선 모든 값에 대해서 row_number 로 각각의 행 번호를 입력
#
#     # 선택한 틴플레이 아이디가 있으면 해당 filter 를 통해 해당 id 의 Rownum을 알아내서 page 변수를 저장합니다.
#     if not page:
#         for obj in videos:
#             if obj.id == teenplayClickId:
#                 page = obj.row_number
#     # 전달받은 페이지 번호가 있으면 row_number을 page로 해서 해당 비디오만 필터링하고, page가 없는경우(최초) videos에 row_number를 page로 저장하여 전달합니다.
#     else:
#         for obj in videos:
#             if obj.id == page:
#                 row_number = page
#
#     # 특정 클럽으로 존재하는 틴플레이 id 를 내림차순으로 정렬하여 id의 value 값을 리스트로 담아서 변수에 저장합니다.
#     teenplay_ids = TeenPlay.enable_objects.filter(club_id=clubId).order_by('-id').values_list('id', flat=True)
#     # 해당 id의 count 를 max count로 저장합니다.
#     max_count = teenplay_ids.count()
#
#     # 현재 보고있는 영상이 좋아요가 눌렸는지 확인하여 변수 대입(idx 번호기 때문에 page-1 형태로 입력)
#     currrent_user_like = TeenPlayLike.objects.filter(member_id=member_id, teenplay_id=teenplay_ids[page-1], status=1)
#
#     # 틴플레이 내서 필요한 정보를 모두 담은 쿼리
#     # teenplay 테이블에 member, club 정보는 담겨져 있기 때문에 집계함수(Count, Exists) , Alias, 집계함수에 대한 조건 필터 사용
#     select_teenplay = \
#         TeenPlay.enable_objects.filter(club_id=clubId, id=teenplay_ids[page-1]).annotate(club_name=F('club__club_name')). \
#             annotate(club_intro=F('club__club_intro')).annotate(club_profile_path=F('club__club_profile_path')). \
#             annotate(teenplay_like=Count('teenplaylike__status', filter=Q(teenplaylike__status=1))). \
#             annotate(member_like=Exists(currrent_user_like)).annotate(
#             tp_all_count=Count('teenplaylike__status', filter=Q(teenplaylike__status=1))). \
#             values('club_id', 'club_name', 'club_intro', 'club_profile_path', 'id', 'video_path', 'teenplay_like',
#                    'member_like', 'tp_all_count')[0]
#     # 해당 틴플레이 딕셔너리에 현재 페이지, 최대 count 입력
#
#     select_teenplay['page'] = page
#     select_teenplay['max_count'] = max_count
#     # 위에서 만든 딕셔너리를 리턴
#     return select_teenplay
#
#
#
# # 클럽에서 틴플레이를 선택 후 스크롤 할 때 데이터를 가져오는 APIView
# class TeenplayClubAPIView(APIView):
#     # get 방식으로 화면에서 필요한 데이터를 전달 받은 후 사용되는 메소드
#     def get(self, request, clubId, teenplayClickId, page):
#         member_id = request.session['member']['id']
#         # get_teenplay_data 메소드를 사용하여 최초 화면에 뿌려질 틴플레이 데이터를 전달하여 해당 메소드의 리턴값을 data 대입
#         data = get_teenplay_data(clubId, page, teenplayClickId, member_id)
#         # data를 전달
#         return Response(data)
#
#
# # 클럽에서 틴플레이를 선택했을 때 틴플레이 재생화면으로 넘어가는 view
# class TeenplayClubView(View):
#     # 틴플레이 재생화면으로 넘어가기 위한 get 메소드 사용
#     def get(self, request, teenplayId):
#         # 세션에 member_id 가 있으면 가져오기 위한 get 함수 사용
#         member_id = request.session['member'].get('id')
#
#         # get_teenplay_data 함수를 호출하여 필요한 데이터를 가져오기
#         teenplayClickId = teenplayId
#         clubId = TeenPlay.enable_objects.get(id=teenplayClickId)
#         clubId = clubId.club_id
#         page = None
#         data = get_teenplay_data(clubId, page, teenplayClickId, member_id)
#         # 함수로 전달받은 결과를 data를 템플릿 컨텍스트에 추가
#         context = {'data': data, 'member_session': {'memberSessionId': member_id}}
#         # 틴플레이 재생화면으로 render를 사용하여 특정 html 로 이동, context 내용도 전달
#         return render(request, 'teenplay/web/teenplay-play-select-web.html', context)
#
#
# # 클럽에서 선택된 틴플레이 좋아요를 눌렀을 떄 사용되는 APIView
# class TeenPlayClubLikeAPIView(APIView):
#     # update 쿼리가 사용되기 때문에 오류 발생 시 DB에 데이터가 수정되지 않도록 transaction.atimic 사용
#     @transaction.atomic
#     def get(self, request, teenplayId, memberSessionId, displayStyle):
#         # member_id, teenplay_id 를 딕셔너리 형태로 생성
#         data = {
#             'member_id': memberSessionId,
#             'teenplay_id': teenplayId
#         }
#
#         # 좋아요 클릭 시 없으면 생성되고 있으면 update 를 하기 위한 get_or_create 함수 사용
#         likeData, checked = TeenPlayLike.objects.get_or_create(**data)
#         # 만약 해당 데이터가 있다면 해당 틴플레이 id 의 total count 를 변수에 대입
#         if checked:
#             totalLikeCount = TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId).count()
#         # 해당 좋아요를 새로 생성했을 때 데이터가 있다면 아래 조건 사용
#         else:
#             # 데이터가 존재하며 현재 좋아요가 클릭되어 있지 않으면
#             if displayStyle != 'none':
#                 # 좋아요 status 를 1로 업데이트 하고 총 카운트를 total count 에 대입한다
#                 TeenPlayLike.objects.filter(status=0, teenplay_id=teenplayId, member_id=memberSessionId).update(status=1, updated_date=timezone.now())
#                 totalLikeCount = TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId).count()
#             else:
#                 # 좋아요 status 를 0으로 업데이트 하고 총 카운트를 total count 에 대입한다
#                 TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId, member_id=memberSessionId).update(status=0, updated_date=timezone.now())
#                 totalLikeCount = TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId).count()
#
#         # 좋아요를 클릭한 결과와 총 count 와 해당 member_id와 display style을 context 에 딕셔너리 형태로 대입하여 응답을 반환
#         context = {
#             'teenplay_id': teenplayId,
#             'member_id': memberSessionId,
#             'display_style': displayStyle,
#             'totalLikeCount': totalLikeCount
#         }
#
#         return Response(context)
#
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
# # 모임에서 틴플레이 선택했을 때 가져오는 것을 사용하는 것으로 예상
# # 틴플레이 좋아요 관련 클래스 생성
# # 데이터를 가져와야 하기 때문에 apiview로 해야한다
# # 만약 해당 맴벅 like테이블에 존재하고 해당 teenplay-id 값을 누른게 있으면 status =1이면 0으로 0이면 1로
# # id 값은 세션의 값을 가져와서 넣어줘야한다
# # 모두 비동기 함수를 사용해야한다.
# # member teenplay , status
#
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# class TeenplayMainListAppView(View):
#     def get(self, request):
#         return render(request, 'teenplay/web/teenplay-play-web.html')
#
#
#

#
# class TeenplayMainListWebView(View):
#     # 틴플레이 화면으로 이동하기 위한 get 함수
#     def get(self, request):
#         # 로그인된 사용자가 있는 경우 해당 로그인 정보를 가져오기 위한 함수 사용
#         member, id = self.session_member_info(request)
#         # 최초 틴플레이 정보를 뿌리기 위한 랜덤 함수
#         ## 만약 맴버가 있으면 해당 id 를 전댈해서 데이터를 가져온 context 파일을 리스트로 가져와야할 듯
#         ## 없을 경우는 뭐 랜덤으로 돌리던 사전 훈련된 모델을 돌리던 해야지
#         context = self.main_random_list(id)
#         # 틴플레이 페이지 이동을 위한 render 함수 사용, context 데이터와 member 데이터를 화면에 가져가기 위해 context 에 딕셔너리 형태로 사용
#         return render(request, 'teenplay/web/teenplay-play-web.html', {'context': context, 'member': member})
#
#     # 틴플레이 좋아요 클릭 시 사용자가 session에 있는지 확인하기 위한 메소드
#     # def session_member_info(self, request):
#     #     # member 객체가 session에 담겨있다면 member_id, member 객체를 member 변수에 담는다.
#     #     if 'member' in request.session and 'id' in request.session['member']:
#     #         id = request.session['member']['id']
#     #         member = request.session['member']
#     #     else:
#     #         # member 객체가 session에 있으면
#     #         if 'member' in request.session:
#     #             # member session에 id 가 있으면 id 를 가져오고 없을 경우 None으로 대입한다
#     #             id = request.session['member'].get('id', None)
#     #         else:
#     #             # 둘다 없을 경우 None 으로 대입한다.
#     #             id = None
#     #             member = None
#     #     # 함수의 결과를 다른곳에서 사용하기 위해 member와 id를 리턴한다.
#     #     return member, id
#
#     # 메인 화면에서 랜덤한 리스트를 5개 뿌리기 위해서 사용하는 함수
#     def main_random_list(self, id):
#         # 틴플레이 중에서 status 가 1인 전체 카운트 확인
#         teenplay_count = TeenPlay.enable_objects.all().count()
#         # 전체 틴플레이 리스트에서 컬럼이 id 에 대한 부분을 쿼리셋 객체에서 list 로 형변환
#         teenplay_id_value = list(TeenPlay.enable_objects.values('id'))
#         # 리스트로 형변환 된 내용중 id에 대한 값을 리스트에 담기 위해 빈 리스트 생성
#         teenplay_id_list = []
#         # 리스트 내부에 딕셔너리 중 value 값만 추출하여 빈 리스트에 추가
#         for teenplay_id_dict in teenplay_id_value:
#             teenplay_id_list.append(teenplay_id_dict['id'])
#
#         # 조회한 결과는 딕셔너리 형태로 확인되기 때문에 해당 내용을 리스트에 담기 위해서 빈 리스트 생성
#         teenplay_list = []
#         # 5번 반복을 하여 내용을 가져오기 위해 range 함수 생성
#         for number in range(5):
#             # 좋아요 수를 담을 빈 딕셔너리 생성
#             like_count = {}
#
#             # 고정된 리스트에서 랜덤한 id 를 1개씩 추출하기 위해 random 내장 라이브러리의 choice 함수 사용
#             radiant_teenplay = choice(teenplay_id_list)
#             # 틴플레이 객체에서 집계함수, 해당 집계함수의 조건(filter) 와 장고 ORM의 특징을 이용해 다른 테이블의 컬럼 결과를 가져와서 teenplay 변수 대입
#             teenplay = TeenPlay.objects.filter(id=radiant_teenplay, status=1).annotate(
#                 likes=Count('teenplaylike__status', filter=Q(teenplaylike__status=1))).values('id', 'video_path',
#                                                                                               'club__id',
#                                                                                               'club__club_name',
#                                                                                               'club__club_intro',
#                                                                                               'club__club_profile_path',
#                                                                                               'club_id', 'likes', 'teenplay_title')
#
#             # 사용자가 해당 틴플레이를 좋아요를 눌렀는지 확인하기 위해서 존재 유무 확인, teenplay_id는 위에서 생성한 랜덤한 id를 동시에 대입
#             member_like = TeenPlayLike.objects.filter(member_id=id, teenplay_id=radiant_teenplay, status=1).exists()
#             # 빈 딕셔너리에 사용자가 좋야요 한 결과를 key value 값으로 대입
#             like_count['like_check'] = member_like
#             # teenplay_like 리스트에 언패킹의 특성을 이용해서 각각의 딕셔너리를 하나씩 추출해서 새로운 딕셔너리로 합치기
#             teenplay_like = {**like_count, **teenplay[0]}
#             # 리스트에 새로 만든 딕셔너리 추가
#             teenplay_list.append(teenplay_like)
#
#         context = teenplay_list
#         # 반복문을 통해 생성된 리스트를 리턴값으로 전달
#         return context
#
# # 틴플레이 리스트를 추가하여 받아올 때 사용하는 API (아래로 스크롤)
# class TeenplayMainListAPIView(APIView):
#     # 해당 url 로 호출을 받으면
#     def get(self, request, slideNumber):
#         teenplay_count = TeenPlay.enable_objects.all().count()
#
#         # 멤버가 현재 세션에 등록되어있는지 확인하는 조건문
#         if 'member' in request.session and 'id' in request.session['member']:
#             id = request.session['member']['id']
#         else:
#             if 'member' in request.session:
#                 id = request.session['member'].get('id', None)
#             else:
#                 id = None
#
#         # 쿼리셋 객체를 리스트로 변경
#         teenplay_id_value = list(TeenPlay.enable_objects.values('id'))
#         teenplay_id_list = []
#         # 변경된 리스트 내부에 있는 딕셔너리 객체를 id 만 추출하여 빈 리스트에 추가
#         for teenplay_id_dict in teenplay_id_value:
#             teenplay_id_list.append(teenplay_id_dict['id'])
#
#
#         teenplay_list = []
#         # 스크롤 진행 시 random 한 id 를 몇개 가져올 지 정하는 반복문
#         random_count = {'random_count': 30}
#         for number in range(random_count['random_count']):
#             like_count = {}
#
#             # 고정된 리스트에서 랜덤한 id 를 1개씩 추출하기 위해 random 내장 라이브러리의 choice 함수 사용
#             radiant_teenplay = choice(teenplay_id_list)
#             # 틴플레이 객체에서 집계함수, 해당 집계함수의 조건(filter) 와 장고 ORM의 특징을 이용해 다른 테이블의 컬럼 결과를 가져와서 teenplay 변수 대입
#             teenplay = TeenPlay.objects.filter(id=radiant_teenplay, status=1).annotate(
#                 likes=Count('teenplaylike__status', filter=Q(teenplaylike__status=1))).values('id', 'video_path','club__club_name','club__club_intro','club__club_profile_path','club_id', 'likes', 'teenplay_title')
#             # 사용자가 해당 틴플레이를 좋아요를 눌렀는지 확인하기 위해서 존재 유무 확인, teenplay_id는 위에서 생성한 랜덤한 id를 동시에 대입
#             member_like = TeenPlayLike.objects.filter(member_id=id, teenplay_id=radiant_teenplay, status=1).exists()
#             # 빈 딕셔너리에 사용자가 좋야요 한 결과를 key value 값으로 대입
#             like_count['like_check'] = member_like
#             # teenplay_like 리스트에 언패킹의 특성을 이용해서 각각의 딕셔너리를 하나씩 추출해서 새로운 딕셔너리로 합치기
#             teenplay_like = {**like_count, **teenplay[0],**random_count}
#             # 리스트에 새로 만든 딕셔너리 추가
#             teenplay_list.append(teenplay_like)
#
#         context = teenplay_list
#         # 반복문을 통해 생성된 리스트를 리턴결과로 전달
#         return Response(context)
#
# # 틴플레이 좋아요를 클릭했을 때 동작하는 APIView
# class TeenPlayLikeAPIView(APIView):
#     # 조회가 아닌 Create, Update 를 사용할 떄 오류로 인한 데이터 변경을 막기 위한 transaction.atomic 데코레이터 사용
#     @transaction.atomic
#     # 화면에서 조회해 온 데이터를 get 방식을 통해 데이터를 받아옴
#     def get(self, request, teenplayId, memberSessionId, displayStyle):
#         # member_id, teenplay_id 를 딕셔너리 형태로 생성
#         data = {
#             'member_id': memberSessionId,
#             'teenplay_id': teenplayId
#         }
#
#         # 좋아요 클릭 시 없으면 생성되고 있으면 update 를 하기 위한 get_or_create 함수 사용
#         likeData, checked = TeenPlayLike.objects.get_or_create(**data)
#         # 만약 해당 데이터가 있다면 해당 틴플레이 id 의 total count 를 변수에 대입
#         if checked:
#             totalLikeCount = TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId).count()
#         # 해당 좋아요를 새로 생성했을 때 데이터가 있다면 아래 조건 사용
#         else:
#             # 데이터가 존재하며 현재 좋아요가 클릭되어 있지 않으면
#             if displayStyle != 'none':
#                 # 좋아요 status 를 1로 업데이트 하고 총 카운트를 total count 에 대입한다
#                 TeenPlayLike.objects.filter(status=0, teenplay_id=teenplayId, member_id=memberSessionId).update(status=1, updated_date=timezone.now())
#                 totalLikeCount = TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId).count()
#             else:
#                 # 좋아요 status 를 0으로 업데이트 하고 총 카운트를 total count 에 대입한다
#                 TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId, member_id=memberSessionId).update(status=0, updated_date=timezone.now())
#                 totalLikeCount = TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId).count()
#
#         # 좋아요를 클릭한 결과와 총 count 와 해당 member_id와 display style을 context 에 딕셔너리 형태로 대입하여 응답을 반환한다.
#         context = {
#             'teenplay_id': teenplayId, # teenplay_id
#             'member_id': memberSessionId,
#             'display_style': displayStyle,
#             'totalLikeCount': totalLikeCount
#         }
#
#         return Response(context)
#
# # 클럽에서 틴플레이를 선택했을 떄 사용되는 함수
# def get_teenplay_data(clubId, page, teenplayClickId, member_id):
#     videos = TeenPlay.enable_objects.filter(club_id=clubId).annotate(row_number=Window(expression=RowNumber(), order_by=F('id').desc()))
#     # 각 인스턴스에 대해 순차 번호(행 번호)를 부여
#     # 우선 모든 값에 대해서 row_number 로 각각의 행 번호를 입력
#
#     # 선택한 틴플레이 아이디가 있으면 해당 filter 를 통해 해당 id 의 Rownum을 알아내서 page 변수를 저장합니다.
#     if not page:
#         for obj in videos:
#             if obj.id == teenplayClickId:
#                 page = obj.row_number
#     # 전달받은 페이지 번호가 있으면 row_number을 page로 해서 해당 비디오만 필터링하고, page가 없는경우(최초) videos에 row_number를 page로 저장하여 전달합니다.
#     else:
#         for obj in videos:
#             if obj.id == page:
#                 row_number = page
#
#     # 특정 클럽으로 존재하는 틴플레이 id 를 내림차순으로 정렬하여 id의 value 값을 리스트로 담아서 변수에 저장합니다.
#     teenplay_ids = TeenPlay.enable_objects.filter(club_id=clubId).order_by('-id').values_list('id', flat=True)
#     # 해당 id의 count 를 max count로 저장합니다.
#     max_count = teenplay_ids.count()
#
#     # 현재 보고있는 영상이 좋아요가 눌렸는지 확인하여 변수 대입(idx 번호기 때문에 page-1 형태로 입력)
#     currrent_user_like = TeenPlayLike.objects.filter(member_id=member_id, teenplay_id=teenplay_ids[page-1], status=1)
#
#     # 틴플레이 내서 필요한 정보를 모두 담은 쿼리
#     # teenplay 테이블에 member, club 정보는 담겨져 있기 때문에 집계함수(Count, Exists) , Alias, 집계함수에 대한 조건 필터 사용
#     select_teenplay = \
#         TeenPlay.enable_objects.filter(club_id=clubId, id=teenplay_ids[page-1]).annotate(club_name=F('club__club_name')). \
#             annotate(club_intro=F('club__club_intro')).annotate(club_profile_path=F('club__club_profile_path')). \
#             annotate(teenplay_like=Count('teenplaylike__status', filter=Q(teenplaylike__status=1))). \
#             annotate(member_like=Exists(currrent_user_like)).annotate(
#             tp_all_count=Count('teenplaylike__status', filter=Q(teenplaylike__status=1))). \
#             values('club_id', 'club_name', 'club_intro', 'club_profile_path', 'id', 'video_path', 'teenplay_like',
#                    'member_like', 'tp_all_count')[0]
#     # 해당 틴플레이 딕셔너리에 현재 페이지, 최대 count 입력
#
#     select_teenplay['page'] = page
#     select_teenplay['max_count'] = max_count
#     # 위에서 만든 딕셔너리를 리턴
#     return select_teenplay
#
#
#
# # 클럽에서 틴플레이를 선택 후 스크롤 할 때 데이터를 가져오는 APIView
# class TeenplayClubAPIView(APIView):
#     # get 방식으로 화면에서 필요한 데이터를 전달 받은 후 사용되는 메소드
#     def get(self, request, clubId, teenplayClickId, page):
#         member_id = request.session['member']['id']
#         # get_teenplay_data 메소드를 사용하여 최초 화면에 뿌려질 틴플레이 데이터를 전달하여 해당 메소드의 리턴값을 data 대입
#         data = get_teenplay_data(clubId, page, teenplayClickId, member_id)
#         # data를 전달
#         return Response(data)
#
#
# # 클럽에서 틴플레이를 선택했을 때 틴플레이 재생화면으로 넘어가는 view
# class TeenplayClubView(View):
#     # 틴플레이 재생화면으로 넘어가기 위한 get 메소드 사용
#     def get(self, request, teenplayId):
#         # 세션에 member_id 가 있으면 가져오기 위한 get 함수 사용
#         member_id = request.session['member'].get('id')
#
#         # get_teenplay_data 함수를 호출하여 필요한 데이터를 가져오기
#         teenplayClickId = teenplayId
#         clubId = TeenPlay.enable_objects.get(id=teenplayClickId)
#         clubId = clubId.club_id
#         page = None
#         data = get_teenplay_data(clubId, page, teenplayClickId, member_id)
#         # 함수로 전달받은 결과를 data를 템플릿 컨텍스트에 추가
#         context = {'data': data, 'member_session': {'memberSessionId': member_id}}
#         # 틴플레이 재생화면으로 render를 사용하여 특정 html 로 이동, context 내용도 전달
#         return render(request, 'teenplay/web/teenplay-play-select-web.html', context)
#
#
# # 클럽에서 선택된 틴플레이 좋아요를 눌렀을 떄 사용되는 APIView
# class TeenPlayClubLikeAPIView(APIView):
#     # update 쿼리가 사용되기 때문에 오류 발생 시 DB에 데이터가 수정되지 않도록 transaction.atimic 사용
#     @transaction.atomic
#     def get(self, request, teenplayId, memberSessionId, displayStyle):
#         # member_id, teenplay_id 를 딕셔너리 형태로 생성
#         data = {
#             'member_id': memberSessionId,
#             'teenplay_id': teenplayId
#         }
#
#         # 좋아요 클릭 시 없으면 생성되고 있으면 update 를 하기 위한 get_or_create 함수 사용
#         likeData, checked = TeenPlayLike.objects.get_or_create(**data)
#         # 만약 해당 데이터가 있다면 해당 틴플레이 id 의 total count 를 변수에 대입
#         if checked:
#             totalLikeCount = TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId).count()
#         # 해당 좋아요를 새로 생성했을 때 데이터가 있다면 아래 조건 사용
#         else:
#             # 데이터가 존재하며 현재 좋아요가 클릭되어 있지 않으면
#             if displayStyle != 'none':
#                 # 좋아요 status 를 1로 업데이트 하고 총 카운트를 total count 에 대입한다
#                 TeenPlayLike.objects.filter(status=0, teenplay_id=teenplayId, member_id=memberSessionId).update(status=1, updated_date=timezone.now())
#                 totalLikeCount = TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId).count()
#             else:
#                 # 좋아요 status 를 0으로 업데이트 하고 총 카운트를 total count 에 대입한다
#                 TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId, member_id=memberSessionId).update(status=0, updated_date=timezone.now())
#                 totalLikeCount = TeenPlayLike.objects.filter(status=1, teenplay_id=teenplayId).count()
#
#         # 좋아요를 클릭한 결과와 총 count 와 해당 member_id와 display style을 context 에 딕셔너리 형태로 대입하여 응답을 반환
#         context = {
#             'teenplay_id': teenplayId,
#             'member_id': memberSessionId,
#             'display_style': displayStyle,
#             'totalLikeCount': totalLikeCount
#         }
#
#         return Response(context)



from datetime import datetime
from django.db.models.functions import Rank, RowNumber
from django.db.models import Count, F, Window, Q, Exists
from django.test import TestCase
from django.utils import timezone
from activity.models import Activity
from django.shortcuts import render, redirect
from django.views import View
from rest_framework.views import APIView
import os.path
from pathlib import Path
import joblib
from rest_framework.response import Response
import numpy as np
from sklearn.preprocessing import Binarizer
from django.db import transaction
from django.db.models.functions import RowNumber
from random import randint, choice
from teenplay.models import TeenPlay, TeenPlayLike
from member.models import MemberFavoriteCategory, Member
from wishlist.models import Wishlist
import sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class TeenplayTest(TestCase):
    pass
    # member_id = 3  # 예제의 member_id 값
    #
    # # 쿼리 1: view_tp_member_favorite_category
    # with connection.cursor() as cursor:
    #     cursor.execute(
    #         "SELECT member_id, member_favorite_category_id FROM view_tp_member_favorite_category WHERE member_id = %s ORDER BY RAND() LIMIT 1",
    #         [member_id]
    #     )
    #     row1 = cursor.fetchone()
    #     columns1 = [col[0] for col in cursor.description]
    #
    # if not row1:
    #     row1 = (member_id, 'default_favorite_category')  # 임의의 값을 할당
    #     columns1 = ['member_id', 'member_favorite_category_id']
    #
    # df1 = pd.DataFrame([row1], columns=columns1) if row1 else pd.DataFrame(columns=columns1)
    #
    # # 쿼리 2: view_tp_member_wishlist_category
    # with connection.cursor() as cursor:
    #     cursor.execute(
    #         "SELECT member_id, wishlist_category_id FROM view_tp_member_wishlist_category WHERE member_id = %s ORDER BY RAND() LIMIT 1",
    #         [member_id]
    #     )
    #     row2 = cursor.fetchone()
    #     columns2 = [col[0] for col in cursor.description]
    #
    # if not row2:
    #     row2 = (member_id, 'default_wishlist_category')  # 임의의 값을 할당
    #     columns2 = ['member_id', 'wishlist_category_id']
    #
    # df2 = pd.DataFrame([row2], columns=columns2) if row2 else pd.DataFrame(columns=columns2)
    #
    # # 쿼리 3: view_tp_teenplay_like_category
    # with connection.cursor() as cursor:
    #     cursor.execute(
    #         "SELECT member_id, club_main_category FROM view_tp_teenplay_like_category WHERE member_id IS NOT NULL GROUP BY club_main_category, member_id ORDER BY COUNT(*) DESC LIMIT 1"
    #     )
    #     row3 = cursor.fetchone()
    #     columns3 = [col[0] for col in cursor.description]
    #
    # if not row3:
    #     row3 = (member_id, 'default_club_main_category')  # 임의의 값을 할당
    #     columns3 = ['member_id', 'club_main_category']
    #
    # df3 = pd.DataFrame([row3], columns=columns3) if row3 else pd.DataFrame(columns=columns3)
    #
    # # 데이터프레임을 member_id를 기준으로 결합
    # df_combined = df1.merge(df2, on='member_id', how='outer').merge(df3, on='member_id', how='outer')
    #
    # print(df_combined.values[0][1:])