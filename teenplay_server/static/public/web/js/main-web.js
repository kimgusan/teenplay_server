const slideBannerBox = document.querySelector(".slide-banner-box");
const bannerPageCount = document.querySelector(".banner-page-count");
let count = 0;

// 첫 화면에서는 첫번째 배너가 보이지만 이 함수가 실행되는 순간 이제 두 번째 배너로 넘어가는 것이다.
function autoSlide() {
    // 이동되는 데 걸리는 시간은 0.5초
    slideBannerBox.style.transition = "transform 0.5s";
    // 마지막 슬라이드일 때
    // 3번 뒤에 1번 배치시킨다.
    // 3번에서 1번으로 슬라이드 진행
    // 0s를 줘서 원래 1번 위치로 이동(슬라이드 효과는 안보임)
    count++;
    if (count === 3) {
        slideBannerBox.style.transform = "translate(-" + 100 * (count + 1) + "%)";
        setTimeout(() => {
            slideBannerBox.style.transition = "transform 0s";
            slideBannerBox.style.transform = "translate(-100%)";
        }, 500);
        count = 0;
        bannerPageCount.innerText = `${count + 1}/3`;
    } else {
        // 처음에는 첫번째 배너가 선택되어 있기 때문에 다음으로 넘어가면서 이전 배너부분의 버튼 색을 돌려준다.
        bannerPageCount.innerText = `${count + 1}/3`;

        // 왼쪽으로 -100 * (count + 1) 만큼 이동한다.
        // 왜 count에 + 1을 한 것인가? : 가장 앞에 6번 배너부터 시작되기 때문이다.
        slideBannerBox.style.transform = "translate(-" + 100 * (count + 1) + "%)";
    }
}

// 무한 반복
let firstDiv = document.createElement("div");
firstDiv.classList.add(`slide-content-wrap`);
let lastDiv = document.createElement("div");
lastDiv.classList.add(`slide-content-wrap`);

// 가장 마지막에 첫번째 배너를 이어 붙인다, 슬라이드 모션이 자연스럽게 1번으로 돌아가게 하기 위함
let firstText = `<div class="slide-content-container">
                    <img class="slide-content-background" alt="대전 청년창업사관학교 14기 전국 스타트업 모집 !" src="/static/public/web/images/main/festival-img01.png" />
                    <div class="slide-text-content">
                        <!-- 해당 축제 링크 필요 -->
                        <a class="slide-text-link" href="" target="_blank">
                            <div class="slide-text-box">
                                <div>
                                    <div class="festival-title">대전 청년창업사관학교 14기 전국 스타트업 모집 !</div>
                                    <div class="festival-contents">
                                        창업지원금에 블루포인트 투자유치 가능성까지
                                        <br />
                                        블루포인트 심사역들이 직접 기획·운영 하는 대전 청년창업사관학교 !
                                    </div>
                                </div>
                                <div class="festival-info">
                                    <div>2024. 01. 15.(월) ~ 2024. 02. 05.(월)</div>
                                    <div>온라인 접수</div>
                                </div>
                            </div>
                        </a>
                    </div>
                </div>`;

firstDiv.innerHTML = firstText;
slideBannerBox.appendChild(firstDiv);

// 가장 첫번째에 마지막 배너를 이어 붙인다, 이전 버튼 클릭 시 슬라이드 모션이 자연스럽게 6번으로 돌아가게 하기 위함
let lastText = `<div class="slide-content-container">
                    <img class="slide-content-background" alt="[모두의특강] ChatGPT Store 오픈!" src="/static/public/web/images/main/festival-img03.jpg" />
                    <div class="slide-text-content">
                        <!-- 해당 축제 링크 필요 -->
                        <a class="slide-text-link" href="" target="_blank">
                            <div class="slide-text-box">
                                <div>
                                    <div class="festival-title">[모두의특강] ChatGPT Store 오픈!</div>
                                    <div class="festival-contents">
                                        직접 사용해 본 유용한 앱은 어떤 것일까?
                                        <br />
                                        그리고 스토어의 미래도 함께 나눠봐요
                                    </div>
                                </div>
                                <div class="festival-info">
                                    <div>2024. 02. 06.(화) 18:00</div>
                                    <div>온라인 행사</div>
                                </div>
                            </div>
                        </a>
                    </div>
                </div>`;

lastDiv.innerHTML = lastText;
slideBannerBox.insertBefore(lastDiv, document.querySelector("div.slide-banner-box div"));

// 첫번째 버튼이 무조건 첫번째 배너이기 때문에 검은색 칠하고 시작
bannerPageCount.innerText = `${count + 1}/3`;

// 첫번째 배너는 3번이니까 왼쪽으로 한 번 밀어서 1번 보이게 함.
slideBannerBox.style.transform = "translate(-100%)";

// 4초마다 슬라이드 이동
let inter = setInterval(autoSlide, 4000);

// 원하는 번호의 배너 보기
// 각 버튼마다 클릭 이벤트 적용

// 이전 버튼, 다음 버튼 구현
const arrows = document.querySelectorAll("div.arrow");

// 버튼을 광클하지 못하게 막아주는 FLAG
let arrowButtonCheck = true;

arrows.forEach((arrow) => {
    // 각 버튼에 click이벤트를 걸어줌.
    arrow.addEventListener("click", () => {
        if (arrowButtonCheck) {
            arrowButtonCheck = false; // 누르자마자 바로 false
            clearInterval(inter); // autoSlide 타이머 제거, 동시에 돌아가면 안됨.
            slideBannerBox.style.transition = "transform 0.5s";
            let arrowType = arrow.classList[0];
            if (arrowType === "banner-left-arrow") {
                // 이전버튼 인지 다음 버튼인지 구분
                count--;
                if (count === -1) {
                    slideBannerBox.style.transform = "translate(0%)";
                    setTimeout(function () {
                        slideBannerBox.style.transition = "transform 0s";
                        slideBannerBox.style.transform = "translate(-300%)";
                    }, 500);
                    count = 2;
                } else {
                    slideBannerBox.style.transform = "translate(-" + 100 * (count + 1) + "%)";
                }
            } else {
                count++;
                if (count === 3) {
                    slideBannerBox.style.transform = "translate(-" + 100 * (count + 1) + "%)";
                    setTimeout(function () {
                        slideBannerBox.style.transition = "transform 0s";
                        slideBannerBox.style.transform = "translate(-100%)";
                    }, 500);
                    count = 0;
                } else {
                    slideBannerBox.style.transform = "translate(-" + 100 * (count + 1) + "%)";
                }
            }
            bannerPageCount.innerText = `${count + 1}/3`;
            inter = setInterval(autoSlide, 4000);
            setTimeout(() => {
                arrowButtonCheck = true;
            }, 500);
        }
    });
});

// 베너 페이지 수 옆 + 클릭 시 베너 모달창 block처리 이벤트
const bannerMoreButton = document.querySelector(".banner-more-button");
const bannerModalContainer = document.querySelector(".banner-modal-container");

bannerMoreButton.addEventListener("click", () => {
    bannerModalContainer.style.display = "block";
});

const bannerCloseContainer = document.querySelector(".banner-close-container");

bannerCloseContainer.addEventListener("click", () => {
    bannerModalContainer.style.display = "none";
});

const bannerModalBox = document.querySelector(".banner-modal-box");

bannerModalBox.addEventListener("click", () => {
    bannerModalContainer.style.display = "none";
});

// 요일과 월을 변환하기 위한 배열
const daysOfWeek = ["일", "월", "화", "수", "목", "금", "토"];
const months = ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"];

function formatDate(dateString) {
    // Date 객체를 생성합니다.
    const date = new Date(dateString);

    // 각 부분을 추출합니다.
    const dayOfWeek = daysOfWeek[date.getDay()]; // 요일
    const day = date.getDate(); // 일
    const month = months[date.getMonth()]; // 월

    // 날짜를 "N j일(l)" 형식으로 구성합니다.
    const formattedDate = `${month} ${day}일(${dayOfWeek})`;

    return formattedDate;
}

const showRecommendedActivities = (activities) => {
    const postLists = document.querySelector("div.post-list.ai-posts");
    let tempLikeBoxes = document.querySelectorAll(".like-box");
    let flag = tempLikeBoxes[0].style.display !== 'none';
    activities.forEach((activity) => {
        postLists.innerHTML += `
            <div class="post-substance">
                <div class="post-info">
                    <div class="post-thumbnail-box">
                        <a class="post-thumbnail-link" href="/activity/detail?id=${activity.id}" ondragstart="return false">
                            <img class="post-thumbnail" alt="활동 썸네일 이미지"
                                src="${activity.thumbnail_path ? '/upload/' + activity.thumbnail_path : '/static/public/web/images/logo/logo8.png'}" 
                            />
                        </a>
                        <div class="like-box" style="${flag ? 'display: block' : 'display: none'}">
                            <input type="hidden" name="activity-id" value="${activity.id}">
                            <button class="like-btn" type="button">
                                <span class="full-heart" style="${activity.is_like ? 'display: block' : 'display: none'}">
                                    <svg data-v-e13ecf0e="" xmlns="http://www.w3.org/2000/svg" class="like-svg" viewBox="0 0 20 20" fill="currentColor">
                                        <path data-v-e13ecf0e="" fill-rule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clip-rule="evenodd"></path>
                                    </svg>
                                </span>
                                <span class="empty-heart" style="${activity.is_like ? 'display: none' : 'display: block'}">
                                    <svg data-v-e13ecf0e="" xmlns="http://www.w3.org/2000/svg" class="unlike-svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path data-v-e13ecf0e="" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path>
                                    </svg>
                                </span>
                            </button>
                        </div>
                    </div>
                    <div class="post-contents">
                        <div class="start-date-place">
                            <div>
                                <span>${formatDate(activity.activity_start)}</span>
                            </div>
                            <div>
                                <span></span>
                            </div>
                        </div>
                        <div class="post-title">
                            <a class="post-title-link" href="/activity/detail?id=${activity.id}" ondragstart="return false">${activity.activity_title}</a>
                        </div>
                        <div class="entry-fee-hits">
                            <div class="hits">
                                <span class="hits-text">관심 ${activity.like_count} | 참가 ${activity.member_count}명</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>`;
    })
    addClickEventToLikeBtns();
}

const getRecommendedActivities = async (callback) => {
    if (!memberId) memberId = 0;
    response = await fetch(`/ai/api/activities/`);
    activities = await response.json();
    if (callback) {
        callback(activities.activities);
    }
}

getRecommendedActivities(showRecommendedActivities);

// 좋아요 선택 시 동작하는 js
const modalWrapCopy = document.querySelector(".club-modal-wrap");
// > 좋아요 선택 시 관심 설정할 때 표시할 부분
const modalLikeContainerCopy = document.querySelector(".club-modal-like-wrap:not(.unlike)");
// > 좋아요 선택 시 관심 해제할 때 표시할 부분
const modalUnlikeContainerCopy = document.querySelector(".club-modal-like-wrap.unlike");


// 좋아요 선택 시 창 내 버튼 클릭 시 모달창 닫기
const modalLikeExitBtnCopy = document.querySelector(".club-modal-like-button");
const modalUnlikeExitBtnCopy = document.querySelector(".club-modal-unlike-button");

function exitModal() {
    modalWrapCopy.style.display = "none";
}

modalLikeExitBtnCopy.addEventListener("click", exitModal);
modalUnlikeExitBtnCopy.addEventListener("click", exitModal);

const getActivityMemberCount = async (activityId) => {
    const response = await fetch(`/activity/members/api?id=${activityId}`);
    const memberCount = await response.json();

    return memberCount;
}

const getActivityLikeCount = async (activityId) => {
    const response = await fetch(`/activity/likes/api?id=${activityId}`);
    const likeCount = await response.json();

    return likeCount;
}

// 좋아요 버튼 클릭 시 svg 변경되는 이벤트

HTMLCollection.prototype.forEach = Array.prototype.forEach;

const addClickEventToLikeBtns = () => {
    const likeBtns = document.querySelectorAll(".like-btn");
    const emptyHeartsCopy = document.querySelectorAll(".empty-heart");
    const fullHeartsCopy = document.querySelectorAll(".full-heart");
    const targetActivityIds = document.querySelectorAll("input[name=activity-id]");
    const hitsText = document.querySelectorAll(".hits-text");
    likeBtns.forEach((likeBtn, i) => {
        likeBtn.addEventListener("click", async () => {
            modalWrapCopy.style.display = "block";
            if (emptyHeartsCopy[i].style.display === "none") {
                await activityLikeCountService.addOrDeleteLike(targetActivityIds[i].value, false);
                modalUnlikeContainerCopy.style.display = "block";
                modalLikeContainerCopy.style.display = "none";
                emptyHeartsCopy[i].style.display = "block";
                fullHeartsCopy[i].style.display = "none";
            } else {
                await activityLikeCountService.addOrDeleteLike(targetActivityIds[i].value, true);
                modalUnlikeContainerCopy.style.display = "none";
                modalLikeContainerCopy.style.display = "block";
                emptyHeartsCopy[i].style.display = "none";
                fullHeartsCopy[i].style.display = "block";
            }
            const likeCount = await getActivityLikeCount(targetActivityIds[i].value);
            const memberCount = await getActivityMemberCount(targetActivityIds[i].value);
            hitsText[i].innerText = `관심 ${likeCount} | 참가 ${memberCount}명`;
        });
    });
}

addClickEventToLikeBtns();

// fetch로 받은 데이터 html에 추가
const showRecommendedClubs = (clubs) => {
    const clubPostLists = document.querySelector("div.post-list.club-ai-posts");
    let text = ``;
    // 24.08.07 [Object object] 로 나타나는 오류가 있어 수정, github 수정
    clubs.forEach((club) => {
        const thumbnailSrc = (club.club_profile_path.club_profile_path && club.club_profile_path.club_profile_path !== ' ') ? `/upload/${club.club_profile_path.club_profile_path}` : '/static/public/web/images/logo/logo1.png';
        text += `
        <div class="post-substance">
            <div class="clup-wrap">
                <div class="clup-container">
                    <a href="/club/detail?id=${club.id}" ondragstart="return false">
                        <div class="clup-profile-box">
                            <div class="clup-profile-thumbnail">
                                <img class="clup-thumbnail-img" alt="모임 썸네일 이미지" src="${thumbnailSrc}" />
                            </div>
                            <div class="badge-gap">
                                <div class="vacuum"></div>
                            </div>
                        </div>
                        <div class="club-info">
                            <input type="hidden" name="club-id" value="${club.id}">
                            <div class="clup-name">${club.club_name.club_name}</div>
                            <div class="clup-activity">
                                ${club.activity_count}
                                <span>개의 활동</span>
                            </div>
                        </div>
                    </a>
                    <div class="clup-service">
        `;
        // 가입 대기자일 경우 가입대기 버튼 활성화 여부
        if (club.is_member != -1 || club.login == 0) {
            text += `
                        <button class="cancel-btn" type="button" style="display: none">
                            <span>가입대기</span>
                        </button>
            `
        } else {
            text += `
                        <button class="cancel-btn" type="button">
                            <span>가입대기</span>
                        </button>
            `
        }
        // 가입한 모임일 경우 가입중 버튼 활성화 여부
        if (club.is_member != 1 || club.login == 0) {
            text += `
                        <div class="clup-btn-box" style="display: none">
                            <button class="join-status" type="button">
                                <svg xmlns="http://www.w3.org/2000/svg" class="check-svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                </svg>
                                <span>가입중</span>
                            </button>
                            <button id="signal-btn" class="signal-on" type="button">
                                <svg xmlns="http://www.w3.org/2000/svg" class="signal-on-svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6zM10 18a3 3 0 01-3-3h6a3 3 0 01-3 3z"></path>
                                </svg>
                                <span>알림 받는 중</span>
                            </button>
                        </div>
            `
        } else {
            text += `
                        <div class="clup-btn-box">
                            <button class="join-status" type="button">
                                <svg xmlns="http://www.w3.org/2000/svg" class="check-svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                </svg>
                                <span>가입중</span>
                            </button>
                            <button id="signal-btn" class="signal-on" type="button">
                                <svg xmlns="http://www.w3.org/2000/svg" class="signal-on-svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6zM10 18a3 3 0 01-3-3h6a3 3 0 01-3 3z"></path>
                                </svg>
                                <span>알림 받는 중</span>
                            </button>
                        </div>
            `
        }
        // 가입되지 않은 경우 가입신청 버튼 활성화 여부
        if (club.is_member != 0 || club.is_manager || club.login == 0) {
            text += `
                        <button class="join-btn" type="button" style="display: none">
                            <span>가입 신청</span>
                        </button>
            `
        } else {
            text += `
                        <button class="join-btn" type="button">
                            <span>가입 신청</span>
                        </button>
            `
        }
        if (!club.is_manager || club.login == 0) {
            text += `
                        <button class="manage-btn" type="button" style="display: none">
                            <span>관리하기</span>
                        </button>
            `
        } else {
            text += `
                        <button class="manage-btn" type="button">
                            <span>관리하기</span>
                        </button>
            `
        }
        text += `
                    </div>
                </div>
            </div>
        </div>
        `;
    })
    clubPostLists.innerHTML =  text;
    postScrollAdd();
    clubBtnAdd();
}


// 인기 모임 추천 데이터 불러오기
const getRecommendedClubs = async (callback) => {
    if (!memberId) memberId = 0;
    response = await fetch(`/ai/api/clubs/`);
    clubs = await response.json();
    if (callback) {
        callback(clubs.clubs);
    }
}

getRecommendedClubs(showRecommendedClubs);

function clubBtnAdd() {
    // 가입 관련 모듈
    const changeClubMemberStatus = async (memberId, clubId) => {
        await fetch(`/club/club-member/api/${memberId}/${clubId}/`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrftoken
                }
            });
    }

    // 가입 대기 클릭 시 모달, 가입상태를 표시하는 이벤트
    const cancelBtns = document.querySelectorAll(".cancel-btn");
    cancelBtns.forEach((btn, i) => {
        btn.addEventListener("click", () => {
            Swal.fire({
                title: "가입신청을 취소하시겠습니까?",
                text: '승인대기 중입니다. 취소하시려면 "신청취소"를 눌러주세요.',
                icon: "warning",
                showCancelButton: true,
                confirmButtonColor: "#3085d6",
                cancelButtonColor: "#d33",
                confirmButtonText: "신청취소",
                cancelButtonText: "닫기",
            }).then(async (result) => {
                if (result.value) {
                    const clubId = clubIds[i].value;
                    await changeClubMemberStatus(memberId, clubId);
                    btn.style.display = "none";
                    joinBtns[i].style.display = "flex";
                    joinCheckBox.style.display = "none";
                    joinMaodalContainer.style.display = "block";
                    joinCancleMent.style.display = "block";
                    joinApplicationMent.style.display = "none";
                    joinGuideMent.innerText = "언제든 다시 신청하실 수 있어요.";
                    continuouslyBtn.style.removeProperty("flex");
                    continuouslyBtn.innerText = "계속 살펴보기";
                    clupPageBtn.style.display = "flex";
                    clupPageBtn.innerText = "모임 상세 바로가기";
                    clupPageBtn.addEventListener("click", () => {
                        joinMaodalContainer.style.display = "none";
                        location.href = `/club/detail?id=${clubId}`;
                    })
                } else if (result.dismiss === "cancel") {
                    return;
                }
            });

        })
    })

    // 가입 신청 클릭 시 모달, 가입상태, 알림을 표시하는 이벤트
    const joinBtns = document.querySelectorAll(".join-btn");
    const joinMaodalContainer = document.querySelector(".join-maodal-container");
    const joinGuideMent = document.querySelector(".join-guide-ment");
    const joinApplicationMent = document.querySelector(".join-application-ment");
    const joinCheckBox = document.querySelector(".join-check-box");
    const clubIds = document.querySelectorAll("input[name=club-id]")

    if (member){
        joinBtns.forEach((joinBtn, i) => {
            joinBtn.addEventListener("click",() => {
                Swal.fire({
                    title: "가입을 신청하시겠습니까?",
                    text: '다시 한번 확인 후 신청버튼을 눌러주세요!',
                    icon: "warning",
                    showCancelButton: true,
                    confirmButtonColor: "#3085d6",
                    cancelButtonColor: "#d33",
                    confirmButtonText: "신청",
                    cancelButtonText: "닫기",
                }).then(async (result) => {
                    if (result.value) {
                        const clubId = clubIds[i].value;
                        await changeClubMemberStatus(memberId, clubId);
                        joinBtn.style.display = "none";
                        cancelBtns[i].style.display = "flex";
                        joinCheckBox.style.display = "flex";
                        joinMaodalContainer.style.display = "block";
                        joinCancleMent.style.display = "none";
                        joinApplicationMent.style.display = "block";
                        joinGuideMent.innerText = "가입 신청이 처리 된다면 알림으로 알려드려요.";
                        continuouslyBtn.style.removeProperty("flex");
                        continuouslyBtn.innerText = "계속 살펴보기";
                        clupPageBtn.style.display = "flex";
                        clupPageBtn.innerText = "모임 상세 바로가기";
                        clupPageBtn.addEventListener("click", () => {
                            joinMaodalContainer.style.display = "none";
                            location.href = `/club/detail?id=${clubId}`;
                        })
                    } else if (result.dismiss === "cancel") {
                        return;
                    }
                });
            });
        });
    }

    const loginMent = document.querySelector(".login-ment");

    // 모임 가입 버튼 클릭 시 로그인 상태가 아니라면 나오는 모달 내용
    if (!member){
        joinBtns.forEach((joinBtn) => {
            joinBtn.addEventListener("click", () => {
                joinMaodalContainer.style.display = "block";
                joinCheckBox.style.display = "none";
                loginMent.style.display = "block";
                joinGuideMent.innerText = "바로 로그인하고, 모임을 가입하세요.\n새롭게 개설되는 활동 안내를 받을 수 있습니다.";
                continuouslyBtn.style.removeProperty("flex");
                continuouslyBtn.innerText = "취소";
                clupPageBtn.style.display = "flex";
                clupPageBtn.innerText = "로그인 하기";
                clupPageBtn.addEventListener("click", () => {
                    joinMaodalContainer.style.display = "none";
                    location.href = `/member/login`;
                })
            });
        });
    }

    // 계속 살펴보기 또는 확인 클릭 시 모달을 닫는 이벤트
    const continuouslyBtn = document.querySelector(".continuously-btn");

    continuouslyBtn.addEventListener("click", () => {
        joinMaodalContainer.style.display = "none";
        joinCancleMent.style.display = "none";
        joinApplicationMent.style.display = "none";
        signalCancleMent.style.display = "none";
        loginMent.style.display = "none";
    });

    // 가입중 클릭 시 모달, 가입 신청을 표기하는 이벤트
    const joinStatuses = document.querySelectorAll(".join-status");
    const joinCancleMent = document.querySelector(".join-cancle-ment");
    const clupPageBtn = document.querySelector(".clup-page-btn");

    joinStatuses.forEach((joinStatus, i) => {
        joinStatus.addEventListener("click",() => {
            Swal.fire({
                    title: "모임을 탈퇴하시겠습니까?",
                    text: '다시 한번 확인 후 탈퇴버튼을 눌러주세요!',
                    icon: "warning",
                    showCancelButton: true,
                    confirmButtonColor: "#3085d6",
                    cancelButtonColor: "#d33",
                    confirmButtonText: "탈퇴",
                    cancelButtonText: "닫기",
            }).then(async (result) => {
                if (result.value) {
                    const clubId = clubIds[i].value;
                    await changeClubMemberStatus(memberId, clubId);
                    joinMaodalContainer.style.display = "block";
                    joinCheckBox.style.display = "none";
                    joinCancleMent.style.display = "block";
                    joinGuideMent.innerText = "모임에서 탈퇴했습니다.\n더이상 새로운 모임 알림 알림을 받을 수 없습니다.";
                    joinStatus.parentElement.style.display = "none";
                    joinStatus.parentElement.nextElementSibling.style.display = "flex";
                    continuouslyBtn.style.flex = "1 1 0%";
                    continuouslyBtn.innerText = "확인";
                    clupPageBtn.style.display = "none";
                } else if (result.dismiss === "cancel") {
                    return;
                }
            });

        });
    });

    // 알림 끄기
    const changeClubAlarmStatus = async (clubId) => {
        await fetch(`/member/club-alarm/api?club-id=${clubId}`);
    }

    // 알림 받는 중 버튼 클릭 시 모달, 제목, 내용 등 바꾸는 이벤트
    const signalBtns = document.querySelectorAll("#signal-btn");
    const signalCancleMent = document.querySelector(".signal-ment");
    let signalCheck = true;

    signalBtns.forEach((signalBtn, i) => {
        signalBtn.addEventListener("click", async () => {
            await changeClubMemberStatus(memberId, clubIds[i]);
            signalBtn.classList.toggle("signal-on");
            signalBtn.classList.toggle("signal-off");
            signalBtn.firstElementChild.classList.toggle("signal-on-svg");
            signalBtn.firstElementChild.classList.toggle("signal-off-svg");

            let spanText = signalBtn.lastElementChild.innerText;
            if (spanText === "알림 받는 중") {
                signalBtn.lastElementChild.innerText = "알림 설정";
            } else {
                signalBtn.lastElementChild.innerText = "알림 받는 중";
            }

            joinMaodalContainer.style.display = "block";

            if (signalCheck) {
                joinCheckBox.style.display = "none";
                signalCancleMent.style.display = "block";
                signalCancleMent.innerText = "모임 알림을 해제했습니다.";
                joinGuideMent.innerText = "더 이상 새로운 모임 알림을 받을 수 없습니다.";
                continuouslyBtn.style.flex = "1 1 0%";
                continuouslyBtn.innerText = "확인";
                clupPageBtn.style.display = "none";
                signalCheck = false;
            } else {
                joinCheckBox.style.display = "flex";
                signalCancleMent.style.display = "block";
                signalCancleMent.innerText = "모임 알림이 설정되었습니다.";
                joinGuideMent.innerText = "가입한 모임에 새로운 소식이 생긴다면 알림으로 알려드려요.";
                continuouslyBtn.style.flex = "1 1 0%";
                continuouslyBtn.innerText = "확인";
                clupPageBtn.style.display = "none";
                signalCheck = true;
            }
        });
    });


    // 관리하기 버튼 클릭 시 모임관리 페이지로 이동
    const manageBtns = document.querySelectorAll(".manage-btn");
    if (manageBtns) {
        manageBtns.forEach((btn, i) => {
            btn.addEventListener("click", () => {
                let clubId = clubIds[i].value;
                location.href = `/member/mypage-club/${clubId}/`;
            })
        })
    }
}