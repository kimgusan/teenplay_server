import os
from pathlib import Path

from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
import joblib
import numpy as np

from activity.models import Activity, ActivityLike, ActivityMember
from activity.serializers import ActivitySerializer
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








