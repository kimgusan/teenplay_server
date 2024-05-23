from datetime import datetime

from django.db import transaction
from django.db.models import F, Q, Count, Exists
from django.shortcuts import render
from django.utils import timezone
from django.views import View
from rest_framework.response import Response
from rest_framework.views import APIView

from teenplay_server.category import Category
from teenplay_server.utils.util.util import check_the_comments
from wishlist.models import Wishlist, WishlistReply, WishlistTag, WishListLike


class WishListView(View):
    # 페이지 이동
    def get(self, request):
        wishlist_id = None
        context = {
            'wishlistId': wishlist_id
        }
        return render(request, 'wishlist/web/wishlist-web.html', context=context)

    def post(self, request):
        wishlist_id = request.POST.get('wishlist-id')
        context = {
            'wishlistId': wishlist_id
        }
        return render(request, 'wishlist/web/wishlist-web.html', context=context)


# 위시리스트 작성
class WishListWriteAPI(APIView):
    def post(self, request):
        data = request.data
        member_id = request.session['member']['id']
        is_private = data['is_private']
        wishlist_content = data['wishlist_content']
        category_id = data['category_id']
        tag_names = data['tag_name']

        # Wishlist 생성
        wishlist = Wishlist.objects.create(member_id=member_id, is_private=is_private, wishlist_content=wishlist_content, category_id=category_id)
        # WishlistTag 확인 및 생성
        for tag_name in tag_names:
            WishlistTag.objects.create(tag_name=tag_name, wishlist_id=wishlist.id)

        return Response('success')


# 위시리스트 리스트
class WishListAPI(APIView):
    def get(self, request, page):
        # id 검사
        if 'member' in request.session and 'id' in request.session['member']:
            id = request.session['member']['id']
        else:
            if 'member' in request.session:
                id = request.session['member'].get('id', None)
            else:
                id = None

        # 페이지 제한 및 필터 적용
        row_count = 3
        offset = (page - 1) * row_count
        limit = page * row_count

        category = request.GET.get('category', '100')
        keyword = request.GET.get('keyword', '없음')

        condition = Q()
        if category != '100':
            condition &= Q(category_id=category)
        if keyword != '없음':
            condition &= Q(wishlisttag__tag_name__contains=keyword)
        # print(condition)

        columns = [
            'wishlist_content',
            'member_id',
            'member_name',
            'category_name',
            'category_id',
            'is_private',
            'created_date',
            'id',
            'member_email',
            'member_path',
            'like_on',
            'like_total',
            'reply_total'
        ]

        wishlists = Wishlist.enabled_objects.filter(condition).annotate(
            member_name=F("member__member_nickname"),
            category_name=F("category__category_name"),
            member_email=F('member__member_email'),
            member_path=F('member__memberprofile__profile_path'),
            like_on=Count('wishlistlike__id', filter=Q(wishlistlike__status=1) & Q(wishlistlike__member_id=id)),
            like_total=Count('wishlistlike__id', filter=Q(wishlistlike__status=1)),
            reply_total=Count('wishlistreply__id', filter=Q(wishlistreply__status=1))

        ).values(*columns).order_by('-id')[offset:limit]

        # 마이페이지에서 넘어왔을 경우
        my_wishlist_id = request.GET.get('wishlist-id', 0)
        if my_wishlist_id and my_wishlist_id != "None":
            my_wishlist = Wishlist.enabled_objects.filter(id=my_wishlist_id).annotate(
                member_name=F("member__member_nickname"),
                category_name=F("category__category_name"),
                member_email=F('member__member_email'),
                member_path=F('member__memberprofile__profile_path'),
                like_on=Count('wishlistlike__id', filter=Q(wishlistlike__status=1) & Q(wishlistlike__member_id=id)),
                like_total=Count('wishlistlike__id', filter=Q(wishlistlike__status=1)),
                reply_total=Count('wishlistreply__id', filter=Q(wishlistreply__status=1))
            ).values(*columns)
            # print(my_wishlist)
            if my_wishlist.exists():
                wishlists = list(wishlists)
                wishlists.insert(0, my_wishlist.first())

        # 태그
        wishlist_ids = [item['id'] for item in wishlists]
        tag_info_by_wishlist = {}

        for wishlist_id in wishlist_ids:
            tag_info = WishlistTag.objects.filter(wishlist_id=wishlist_id).values_list('tag_name', flat=True)
            tag_info_by_wishlist[wishlist_id] = list(tag_info)

        data = {
            'wishlists': wishlists,
            'tags': tag_info_by_wishlist,
            'my_wishlist': my_wishlist_id
        }

        return Response(data)


# 위시리스트 수정/삭제
class WishListActionAPI(APIView):
    def post(self, request, wishlist_id):
        wishlist = Wishlist.objects.get(id=wishlist_id)
        wishlist.status = 0

        wishlist.save(update_fields=['status'])

        return Response('success')

    def patch(self, request, wishlist_id):
        datas = request.data
        # print(datas)
        # 위시리스트 정보 처리
        new_wishlist = datas.get('new_wishlist')
        # print(new_wishlist)
        # print(new_wishlist_id)
        new_wishlist_content = new_wishlist.get('wishlist_content')
        new_category_id = new_wishlist.get('category_id')
        new_is_private = new_wishlist.get('is_private')
        updated_date = timezone.now()

        wishlist = Wishlist.objects.get(id=wishlist_id)
        wishlist.wishlist_content = new_wishlist_content
        wishlist.category_id = new_category_id
        wishlist.is_private = new_is_private
        wishlist.updated_date = updated_date

        wishlist.save(update_fields=['wishlist_content', 'category_id', 'is_private', 'updated_date'])

        # 태그 정보 처리
        new_tag_name = new_wishlist.get('tag_name')
        # print(new_tag_name)
        for tag_name in new_tag_name:
            wishlist_update_object,checked = WishlistTag.objects.get_or_create(wishlist_id=wishlist_id, tag_name=tag_name)
            # print(tag_name)
        #
        # if checked is False:
        #     WishlistTag.objects.filter(wishlist_id=wishlist_id).update(status=0)


        return Response('success')


# 댓글 작성
class ReplyWriteAPI(APIView):
    def post(self, request):
        data = request.data
        data = {
            'reply_content': data['reply_content'],
            'wishlist_id': data['wishlist_id'],
            'member_id': request.session['member']['id']
        }

        result = check_the_comments(data['reply_content'])

        if result == 'profanity':
            return Response(result)

        WishlistReply.objects.create(**data)
        return Response('success')


# 댓글 리스트
class ReplyListAPI(APIView):
    def get(self, request):
        wishlist_id = request.GET.get('id')

        columns = [
            'wishlist_id',
            'member_id',
            'member_name',
            'reply_content',
            'created_date',
            'updated_date',
            'member_email',
            'id',
            'member_path'
        ]

        replies = WishlistReply.enabled_objects.filter(wishlist_id=wishlist_id).annotate(member_name=F("member__member_nickname"),member_email=F('member__member_email'), member_path=F('member__memberprofile__profile_path')).values(*columns)

        return Response(replies)


# 댓글 수정/삭제
class ReplyActionAPI(APIView):
    def post(self, request, reply_id):
        reply = WishlistReply.objects.get(id=reply_id)
        # print(reply)
        reply.status = 0

        reply.save(update_fields=['status'])

        return Response('success')

    def patch(self, request, reply_id):
        data = request.data
        reply_content = data['reply_content']

        result = check_the_comments(data['reply_content'])

        if result == 'profanity':
            return Response(result)

        updated_date = timezone.now()

        reply = WishlistReply.objects.get(id=reply_id)
        reply.reply_content = reply_content
        reply.updated_date = updated_date

        reply.save(update_fields=['reply_content', 'updated_date'])

        return Response(reply_content)


# 위시리스트 게시글 좋아요
class WishlistLikeAPIView(APIView):
    @transaction.atomic
    def get(self, request, wishlist_id, memberId, displayStyle):

        data = {
            'member_id': memberId,
            'wishlist_id': wishlist_id
        }

        likeData, checked = WishListLike.objects.get_or_create(**data)
        if checked:
            totalLikeCount = WishListLike.objects.filter(status=1, wishlist_id=wishlist_id).count()
        else:
            if displayStyle== 'none':
                WishListLike.objects.filter(status=0, wishlist_id=wishlist_id, member_id=memberId).update(status=1, updated_date=timezone.now())
                totalLikeCount = WishListLike.objects.filter(status=1, wishlist_id=wishlist_id).count()
            else:
                WishListLike.objects.filter(status=1, wishlist_id=wishlist_id, member_id=memberId).update(status=0, updated_date=timezone.now())
                totalLikeCount = WishListLike.objects.filter(status=1, wishlist_id=wishlist_id).count()

        context = {
            'wishlist_id': wishlist_id,
            'member_id': memberId,
            'display_style': displayStyle,
            'totalLikeCount': totalLikeCount
        }

        return Response(context)