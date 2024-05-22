### member/models.py 입니다 마이그레이션 드루갑니다!

from django.db import models

from member.managers import MemberManager
from teenplay_server.category import Category

from teenplay_server.period import Period


class Member(Period):
    GENDER_STATUS = [
        (0, '선택안함'),
        (1, '남성'),
        (2, '여성')
    ]

    MEMBER_STATUS = [
        (-1, '정지'),
        (0, '탈퇴'),
        (1, '활동중')
    ]

    member_email = models.TextField(blank=False, null=False)
    member_nickname = models.TextField(blank=False, null=False)
    member_phone = models.TextField(null=True)
    member_address = models.TextField(null=True)
    # 0: 선택안함, 1: 남성, 2: 여성
    member_gender = models.SmallIntegerField(choices=GENDER_STATUS, default=0, null=True)
    # 출생연도
    member_birth = models.IntegerField(null=True)
    # 0: 미동의, 1: 동의
    member_marketing_agree = models.BooleanField(default=0, null=True)
    # 0: 미동의, 1: 동의
    member_privacy_agree = models.BooleanField(default=0, null=True)
    # -1: 정지, 0: 탈퇴, 1: 활동중
    status = models.SmallIntegerField(choices=MEMBER_STATUS, default=1, null=False, blank=False)
    # google, kakao, naver
    member_type = models.CharField(max_length=10, null=False, blank=False)
    # 키워드 1~3
    member_keyword1 = models.TextField(null=True)
    member_keyword2 = models.TextField(null=True)
    member_keyword3 = models.TextField(null=True)
    # 메인페이지 추천활동 관련 ai 모델 필드
    member_recommended_activity_model = models.TextField(null=True)
    # 틴플레이 추천 관련 ai 모델 필드
    member_recommended_teenplay_model = models.TextField(null=True)
    # 모임 추천 관련 ai 모델 필드
    member_recommended_club_model = models.TextField(null=True)

    objects = models.Manager()
    enabled_objects = MemberManager()

    class Meta:
        db_table = 'tbl_member'


class MemberProfile(Period):
    member = models.ForeignKey(Member, null=False, blank=False, on_delete=models.PROTECT)
    # 프사
    profile_path = models.ImageField(null=False, blank=False, upload_to='member/%Y/%m/%d')
    # 0: 프사 없음, 1: 프사 있음
    status = models.BooleanField(null=False, blank=False, default=1)

    objects = models.Manager()
    enabled_objects = MemberManager()

    class Meta:
        db_table = 'tbl_member_profile'

class AdminAccount(Period):
    admin_id = models.TextField(blank=False, null=False)
    admin_password = models.TextField(blank=False, null=False)
    admin_name = models.TextField(blank=False, null=False)

    class Meta:
        db_table = 'tbl_admin_account'


class MemberFavoriteCategory(Period):
    member = models.ForeignKey(Member, blank=False, null=False, on_delete=models.PROTECT)
    category = models.ForeignKey(Category, blank=False, null=False, on_delete=models.PROTECT)
    # 0: 삭제
    status = models.BooleanField(null=False, blank=False, default=1)

    class Meta:
        db_table = 'tbl_member_favorite_category'


class MemberDeleteReason(Period):
    delete_reason = models.SmallIntegerField(blank=False, null=False)
    delete_text = models.TextField(null=True)

    class Meta:
        db_table = 'tbl_member_delete_reason'