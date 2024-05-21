from django.db.models import Q, Count
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views import View
from rest_framework.response import Response
from rest_framework.views import APIView

from activity.models import Activity, ActivityLike
from club.models import Club, ClubMember
from notice.models import Notice
from search.models import RecentSearch


class MainView(View):
    # 메인페이지로 이동하기 위한 view입니다.
    def get(self, request):
        # 현재 로그인한 사용자가 있다면 세션으로부터 받아옵니다.
        member = request.session.get('member')

        # 사용자의 id를 담을 변수의 값을 0으로 초기화합니다.
        member_id = 0

        # 세션에서 get()을 통해 가져온 멤버 객체가 None이 아니라면, 즉 로그인한 상태라면
        if member is not None:
            # 사용자의 id를 저장합니다.
            member_id = member.get('id')

        # 인기 활동 목록에 표시할 활동들을 가져옵니다.
        # 이때 annotate()와 Count()를 활용하여 활동 참가자 수가 많은 순으로 정렬하며, 8개만 조회합니다.
        popular_activities = Activity.enabled_objects\
                                 .annotate(member_count=Count('activitymember')).order_by('-member_count')[:8]

        for activity in popular_activities:
            # 각 인기활동마다 관심활동으로 등록한 사용자의 수를 집계하여 담아줍니다.
            likes = ActivityLike.enabled_objects.filter(activity_id=activity.id).count()
            activity.like_count = likes
            # 각 인기활동마다 현재 로그인한 사용자가 관심활동으로 등록했는지 여부를 조회하여 담아줍니다.
            # 이때 로그인한 상태가 아니라면 member_id가 0으로 초기화되어있으므로 일치하는 데이터가 없어 False가 담기게 됩니다.
            # 화면에서 로그아웃 상태 시 관심활동 등록/삭제 버튼인 하트 아이콘이 아예 표시되지 않도록 되어 있습니다.
            is_like = ActivityLike.enabled_objects.filter(activity_id=activity.id, member_id=member_id).exists()
            activity.is_like = is_like

        # 최신 활동 목록에 표시할 활동들을 가져옵니다.
        # 이때 annotate()와 Count()를 활용하여 활동 참가자 수를 집계하며, 최신순으로 8개만 조회합니다.
        new_activities = Activity.enabled_objects.\
                             annotate(member_count=Count('activitymember')).order_by('-id')[:8]

        for activity in new_activities:
            # 각 활동마다 관심활동으로 등록한 사용자의 수를 집계하여 담아줍니다.
            likes = ActivityLike.enabled_objects.filter(activity_id=activity.id).count()
            activity.like_count = likes
            # 인기활동과 동일한 방식으로 관심활동 여부를 담아줍니다.
            is_like = ActivityLike.enabled_objects.filter(activity_id=activity.id, member_id=member_id).exists()
            activity.is_like = is_like

        # # AI 추천 부분은 추후에 구현하고자 현재는 최신순으로 정렬합니다.
        # # 나머지 로직은 인기활동 및 최신활동과 동일합니다.
        # recommend_activities = Activity.enabled_objects\
        #                            .annotate(member_count=Count('activitymember')).order_by('-id')[:8]
        #
        # for activity in recommend_activities:
        #     # 각 활동마다 관심활동으로 등록한 사용자의 수를 집계하여 담아줍니다.
        #     likes = ActivityLike.enabled_objects.filter(activity_id=activity.id).count()
        #     activity.like_count = likes
        #     is_like = ActivityLike.enabled_objects.filter(activity_id=activity.id, member_id=member_id).exists()
        #     activity.is_like = is_like

        # 인기 모임목록을 구성원 수가 많은 순으로 정렬하여 최대 8개까지 가져옵니다.
        popular_clubs = Club.enabled_objects.annotate(member_count=Count('clubmember')).order_by('-member_count')[:8]

        # 각 인기 모임마다 반복문으로 접근합니다.
        for popular_club in popular_clubs:
            # 각 모임의 id를 가져옵니다.
            club_id = popular_club.id
            # 해당 id를 통해 해당 모임에서 개설한 활동의 개수를 세어 저장합니다.
            activity_count = Activity.enabled_objects.filter(club_id=club_id).count()
            # 각 모임 객체의 activity_count 필드에 활동 개수를 담아줍니다.
            popular_club.activity_count = activity_count
            # 각 모임마다 현재 로그인한 사용자가 모임장인지 여부를 담아줍니다.
            # 로그아웃 상태일 경우 member_id가 0으로 초기화되어있으므로 False가 담깁니다.
            popular_club.is_manager = popular_club.member_id == member_id
            # 각 모임마다 현재 로그인한 사용자가 구성원인지 여부를 담아주기 위한 부분입니다.
            # 로그아웃 상태일 경우 동일한 방식으로 False가 담깁니다.
            is_member = ClubMember.objects.filter(club_id=club_id, member_id=member_id)
            # 만약 모임구성원 테이블(tbl_club_member)에 해당 모임id 및 해당 멤버id와 일치하는 정보가 있다면
            if is_member.exists():
                # first()를 통해 해당 객체를 가져옵니다.
                is_member = is_member.first()
                # is_member에 status를 넣어줍니다.
                # 1일 경우 구성원이며 0일 경우 soft delete로 인해, 혹은 탈퇴로 인해 구성원이 아닌 경우입니다.
                is_member = is_member.status
            # 테이블에 존재하지 않는다면 신청 내역이 존재하지 않으므로 구성원이 아닌 경우입니다.
            else:
                is_member = 0
            # 각 인기 모임마다 구성원 여부를 is_member 필드에 담아줍니다.
            popular_club.is_member = is_member

        # 위에서 정리한 정보들을 화면에 전달하기 위해 context에 담아줍니다.
        context = {
            'popular_activities': popular_activities,
            'new_activities': new_activities,
            # 'recommend_activities': recommend_activities,
            'popular_clubs': popular_clubs
        }

        return render(request, 'main/web/main-web.html', context=context)


class FooterNoticeLatestAPI(APIView):
    # 푸터에서 최신 공지사항 1개를 표시하기 위한 REST API입니다.
    def get(self, request):
        # 공지사항 중 soft delete 되지 않은(status=True) 공지사항을 values()를 통해 딕셔너리 형태로 조회합니다.
        notices = Notice.objects.filter(status=True).values()[:1]

        # 해당 Queryset 객체를 딕셔너리 형태로 Response에 담아 응답합니다.
        notices = {
            'notices': notices
        }
        return Response(notices)


class RecentSearchKeywordAPI(APIView):
    # 최근 검색어를 6개까지 조회하여 응답하는 REST API입니다.
    def get(self, request):
        # 로그아웃 상태일 경우 최근 검색어가 표시되지 않기 때문에 로그인 상태를 가정합니다.
        # 현재 로그인한 사용자의 id를 세션에서 받아옵니다.
        member_id = request.session.get('member').get('id')

        # 해당 id를 기준으로 최근 검색어에서 soft delete되지 않은 검색어를 최신순으로 6개까지
        # values()를 통해 딕셔너리로 조회한 후 해당 Queryset을 list로 형변환합니다.
        keywords = list(RecentSearch.enabled_objects.filter(member_id=member_id).values().order_by('-updated_date')[:6])

        return Response(keywords)

    # 헤더 검색창에서 최근 검색어를 삭제했을 때 호출되는 REST API입니다.
    def delete(self, request):
        # 삭제 시마다 해당 최근 검색어의 id를 fetch 경로 내에 쿼리스트링으로 받아옵니다.
        id = request.data.get('id')

        # 해당 id에 맞는 최근 검색어를 filter()를 통해 조회합니다.
        recent_search = RecentSearch.enabled_objects.filter(id=id)

        # enabled_objects를 사용했으므로 status가 1인 최근 검색어(soft delete되지 않은 검색어)만 조회됩니다.
        # 이때 filter()의 반환값에서 exists()를 통해 존재 여부를 판단합니다.
        if recent_search.exists():
            # 존재한다면 first()를 통해 객체를 가져옵니다.
            recent_search = recent_search.first()
            # soft delete를 위해 status를 0으로 변경한 후 save()를 통해 update합니다.
            recent_search.status = 0
            recent_search.updated_date = timezone.now()
            recent_search.save(update_fields=['status', 'updated_date'])

        return Response("success")


class RecentSearchKeywordDeleteAllAPI(APIView):
    # 헤더의 검색창에서 최근검색어 전체 삭제를 클릭했을 때 호출하는 REST API입니다.
    def get(self, request):
        # 최근 검색어 개별 삭제와 동일한 방식으로 세션에서 사용자의 id를 받아옵니다.
        member_id = request.session.get('member').get('id')

        # 나머지는 모두 동일한 로직입니다.
        recent_searches = RecentSearch.enabled_objects.filter(member_id=member_id)
        for recent_search in recent_searches:
            recent_search.status = 0
            recent_search.updated_date = timezone.now()
            recent_search.save(update_fields=['status', 'updated_date'])

        return Response("success")


class RealTimeSearchAPI(APIView):
    # 헤더의 검색창에서 사용자가 검색어를 입력할 때마다(keyup 이벤트 발생 시) 호출되는 REST API입니다.
    def get(self, request):
        # 검색창 input 태그의 value를 fetch 요청 경로에서 쿼리스트링으로 받아옵니다.
        keyword = request.GET.get('keyword')

        # 헤더의 검색창에서 실시간 검색 결과를 보여줄 때에는 활동 제목을 대상으로만 검색하며,
        # 이때 대소문자 구분 없이 __icontains로 검색어 포함 여부를 조건으로 추가합니다.
        condition = Q(activity_title__icontains=keyword)

        # 이때 실시간 검색결과의 해당 활동 제목을 누를 시 활동 상세 페이지로 이동할 수 있도록 활동 id와,
        # 검색 결과를 표시하기 위한 활동 제목을 values()에 컬럼 이름으로 전달하여 조회하며, 최대 5개까지 조회합니다.
        found_activities = list(Activity.enabled_objects.filter(condition).values('id', 'activity_title')[:5])

        return Response(found_activities)
