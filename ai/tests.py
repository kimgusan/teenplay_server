import os
from pathlib import Path

from django.test import TestCase
from django.utils import timezone

from member.models import Member, MemberFavoriteCategory
import random
from tqdm import tqdm
import joblib
import sklearn


class AiTests(TestCase):
    # member = Member.enabled_objects.get(id=13)
    # member_address = member.member_address
    #
    # if member_address is None or member_address.strip() == '':
    #     print('dd')
    # else:
    #     print('no')

    # keywords = {
    #     1: ['공예', 'DIY', '수집', '독서', '원예', '사진 촬영', '낚시', '요리', '그림 그리기', '보드게임'],
    #     2: ['미술관', '전시회', '공연', '연극', '오페라', '클래식 음악', '영화 감상', '문학', '조각', '건축'],
    #     3: ['헬스', '요가', '필라테스', '등산', '달리기', '자전거', '수영', '클라이밍', '축구', '농구'],
    #     4: ['미식', '와인 시음', '커피 테이스팅', '디저트 만들기', '베이킹', '쿠킹 클래스', '술집 탐방', '페어링', '한식 요리', '세계 음식'],
    #     5: ['배낭여행', '국내 여행', '해외 여행', '캠핑', '트레킹', '로드트립', '힐링 여행', '여행 사진', '여행 계획', '숙소 추천'],
    #     6: ['독서 모임', '강연', '세미나', '온라인 코스', '명상', '리더십', '시간 관리', '목표 설정', '심리학', '인생 코칭'],
    #     7: ['지역 모임', '커뮤니티', '친목회', '동네 카페', '반려동물 산책', '마을 축제', '지역 행사', '동호회', '프리마켓', '이웃 사귀기'],
    #     8: ['데이트', '미팅', '블라인드 데이트', '매칭 이벤트', '소셜 파티', '커플 여행', '로맨스', '연애 상담', '사랑 이야기', '소개팅 팁'],
    #     9: ['투자', '주식', '펀드', '부동산', '재무 계획', '예금', '절약', '금융 상품', '비트코인', '경제 뉴스'],
    #     10: ['영어 회화', '스페인어', '일본어', '중국어', '프랑스어', '독일어', '이탈리아어', '러시아어', '언어 교환', '번역'],
    #     11: ['독서실', '학습법', '자격증 준비', '그룹 스터디', '시험 대비', '논문 작성', '온라인 강의', '집중력 향상', '문제 풀이', '학습 계획'],
    #     12: ['전통 축제', '음악 페스티벌', '문화 행사', '푸드 페스티벌', '연극 축제', '지역 축제', '불꽃놀이'],
    #     13: ['없음', '그 외', '그외']
    # }

    # model_path = os.path.join(Path(__file__).resolve().parent, 'ai/club.pkl')
    # model = joblib.load(model_path)
    #
    # members = list(Member.objects.filter(id__lt=19))
    # for member in tqdm(members):
    #     member_model_path = f'ai/2024/05/22/club_model{member.id}.pkl'
    #     os.makedirs(os.path.dirname(member_model_path), exist_ok=True)
    #     joblib.dump(model, member_model_path)
    #     member.member_recommended_club_model = member_model_path
    #     member.save(update_fields=['member_recommended_club_model'])
        # member_favorite_categories = list(MemberFavoriteCategory.objects.filter(status=1, member=member))
        # chosen_categories = []
        # for category in member_favorite_categories:
        #     chosen_categories += keywords[category.category_id]
        # if not chosen_categories:
        #     for c_ks in keywords.values():
        #         chosen_categories += c_ks
        # member_keywords = random.sample(chosen_categories, k=3)
        # member.member_keyword1 = member_keywords[0]
        # member.member_keyword2 = member_keywords[1]
        # member.member_keyword3 = member_keywords[2]
        # member.updated_date = timezone.now()
        # member.save(update_fields=['member_keyword1', 'member_keyword2', 'member_keyword3', 'updated_date'])
        pass