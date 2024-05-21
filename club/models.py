import datetime
import math
import os

from django.db import models

from club.managers import ClubManager, ClubMemberManager, ClubPostManager, ClubPostReplyManager
from member.models import Member
from teenplay_server.category import Category
from teenplay_server.models import Region
from teenplay_server.period import Period


class Club(Period):
    club_name = models.TextField(null=False, blank=False)
    club_intro = models.TextField(null=False, blank=False)
    club_info = models.TextField(null=True)
    member = models.ForeignKey(Member, null=False, blank=False, on_delete=models.PROTECT)
    club_profile_path = models.ImageField(upload_to='club/%Y/%m/%d')
    club_banner_path = models.ImageField(upload_to='club/%Y/%m/%d')
    club_main_category = models.ForeignKey(Category, null=False, blank=False, default=1, on_delete=models.PROTECT)
    club_region = models.ForeignKey(Region, null=False, blank=False, default=1, on_delete=models.PROTECT)
    # 0: 삭제
    status = models.BooleanField(default=1, null=False)

    # 이거 추가했습니다.
    objects = models.Manager()
    enabled_objects = ClubManager()

    class Meta:
        db_table = 'tbl_club'

    def get_absolute_url(self):
        return f'/club/detail/?id={self.id}'


class ClubMember(Period):
    CLUB_MEMBER_STATUS = [
        (-1, '가입대기'),
        (0, '탈퇴'),
        (1, '가입중')
    ]

    club = models.ForeignKey(Club, null=False, blank=False, on_delete=models.PROTECT)
    member = models.ForeignKey(Member, null=False, blank=False, on_delete=models.PROTECT)
    # 0: 알림끄기, 1: 알림켜기
    alarm_status = models.BooleanField(default=1, null=False, blank=False)
    # -1: 가입대기, 0: 탈퇴, 1: 가입중
    status = models.SmallIntegerField(choices=CLUB_MEMBER_STATUS, default=-1, null=False, blank=False)

    enabled_objects = ClubMemberManager()
    objects = models.Manager()

    class Meta:
        db_table = 'tbl_club_member'


class ClubNotice(Period):
    club = models.ForeignKey(Club, null=False, blank=False, on_delete=models.PROTECT)
    notice_title = models.TextField(null=False, blank=False)
    notice_content = models.TextField(null=False, blank=False)
    # 0: 삭제, 1: 게시중
    status = models.BooleanField(default=1, null=False, blank=False)

    class Meta:
        db_table = 'tbl_club_notice'


class ClubPost(Period):
    club = models.ForeignKey(Club, null=False, blank=False, on_delete=models.PROTECT)
    post_title = models.TextField(null=False, blank=False)
    post_content = models.TextField(null=False, blank=False)
    image_path = models.ImageField(upload_to='club_post/%Y/%m/%d')
    view_count = models.IntegerField(default=0)
    # 0: 삭제, 1: 게시중
    status = models.BooleanField(default=1, null=False, blank=False)

    enabled_objects = ClubPostManager()
    objects = models.Manager()

    class Meta:
        db_table = 'tbl_club_post'

    def get_absolute_url(self):
        return f'/club/pr-post-detail/?id={self.id}'

    def change_date_format(self):
        now = datetime.datetime.now()
        create_date = self.created_date
        gap = math.floor((now - create_date).seconds / 60)

        if gap < 1:
            return "방금 전"

        if gap < 60:
            return f"{gap}분 전"

        gap = math.floor(gap / 60)

        if gap < 24:
            return f"{gap}시간 전"

        gap = math.floor(gap / 24)

        if gap < 31:
            return f"{gap}일 전"

        gap = math.floor(gap / 31)

        if gap < 12:
            return f"{gap}개월 전"

        gap = math.floor(gap / 12)
        return f"{gap}년 전"

    def image_delete(self):
        os.remove(self.image_path.path)


class ClubPostReply(Period):
    club_post = models.ForeignKey(ClubPost, null=False, blank=False, on_delete=models.PROTECT)
    member = models.ForeignKey(Member, null=False, blank=False, on_delete=models.PROTECT)
    reply_content = models.TextField(null=False, blank=False)
    # 0: 삭제, 1: 게시중
    status = models.BooleanField(default=1, null=False, blank=False)

    enabled_objects = ClubPostReplyManager()
    objects = models.Manager()

    class Meta:
        db_table = 'tbl_club_post_reply'


class ClubCategory(Period):
    category = models.ForeignKey(Category, null=False, blank=False, on_delete=models.PROTECT)
    club = models.ForeignKey(Club, null=False, blank=False, on_delete=models.PROTECT)
    # 0: 삭제, 1: 유효
    status = models.BooleanField(default=1, null=False, blank=False)

    class Meta:
        db_table = 'tbl_club_category'
