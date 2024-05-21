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


class TeenplayTest(TestCase):
    pass
#     # def get_user_features(self, member_id):
#     # 로그인 했을 때 사용자가 가지고 있는
#     # 1. 사용자의 카테고리 id
#     member_like_random_category = MemberFavoriteCategory.objects.filter(member_id=2).order_by('?').values('category_id').first()
#     # 2. 위시리스트의 카테고리 id
#     wishlist_category = Wishlist.objects.filter(member_id=2).values('category_id').first()
#     # 3. 최고 좋아요 클럽의 카테고리 id
#     teenplay_like_most_category = (TeenPlayLike.objects.filter(member_id=2)
#                                    .annotate(category_id=F('teenplay__club__club_main_category_id'))
#                                    .values('category_id')
#                                    .annotate(category_count=Count('category_id'))
#                                    .order_by('-category_count')
#                                    .first())
#
#     # if wishlist_category is None:
#     #     wishlist_category = {'category_id': teenplay_like_most_category['category_id']}
#     #
#     # if teenplay_like_most_category is None:
#     #     teenplay_like_most_category = {'category_id': teenplay_like_most_category['category_id']}
#
#     # print(teenplay_like_most_category['category_id'])
#     # print(wishlist_category['category_id'])
#     # print(member_like_random_category['category_id'])
#     # return teenplay_like_most_category['category_id'], member_like_random_category['category_id'], wishlist_category['category_id']
#
#     # 피클파일 사용
#     data_list= [teenplay_like_most_category['category_id'], member_like_random_category['category_id'],wishlist_category['category_id']]
#     datas = np.array(data_list).astype('int32').reshape(1, -1)
#     model_path = Path(__file__).resolve().parent.parent/'ai'/'ai'/'rfc_default_model.pkl'
#     model = joblib.load(model_path)
#     prediction_proba = model.predict_proba(datas)
#     top_3_indices = np.argsort(prediction_proba[0])[-3:][::-1]
#     top_3_classes = model.classes_[top_3_indices]
#
#     # 특정 조건을 만족하는 객체들을 최대 30개까지 필터링하여 가져오기
#     teenplays = TeenPlay.enable_objects.filter(club__club_main_category_id__in=[4, 5, 6])[:30]
#
#     # 결과 출력
#
#     teenplay_count =TeenPlay.enable_objects.filter(club__club_main_category_id__in=[4,5,6]).count()
#     # print(teenplay_count)
#     # print(list(TeenPlay.enable_objects.values('id')))
#     # print(TeenPlay.enable_objects.filter(club__club_main_category_id__in=[4,5,6]).values('id'))
# # =========================================================================================================================
#     teenplay_id_value = list(TeenPlay.enable_objects.filter(club__club_main_category_id__in=[12,13,12]).values('id'))
#     # 리스트로 형변환 된 내용중 id에 대한 값을 리스트에 담기 위해 빈 리스트 생성
#     teenplay_id_list = []
#     # 리스트 내부에 딕셔너리 중 value 값만 추출하여 빈 리스트에 추가
#     for teenplay_id_dict in teenplay_id_value:
#         teenplay_id_list.append(teenplay_id_dict['id'])
