const club = club_list[0];

// 공유하기 버튼 클릭 시 모달창으로 클립보드에 url 복사
const shareBtn = document.getElementById("share");
function clipCopy() {
    let dummy = document.createElement("textarea");
    document.body.appendChild(dummy);
    dummy.value = window.document.location.href;
    dummy.select();
    document.execCommand("copy");
    document.body.removeChild(dummy);
    Swal.fire("URL 복사 완료", "주소가 클립보드에 복사되었습니다. <br> 원하는 곳에 붙여넣기 해주세요.", "success");
}
shareBtn.addEventListener("click", clipCopy);

// 상단 모임 버튼
// 모임 이름 받아와서 넣어야함(아래는 예시)
const clubName = document.querySelector(".club-detail-name").innerText;

// 관리하기 버튼 클릭 시 모임 관리 페이지로 이동
const manageBtnEvent = () => {
    document.getElementById("manage").addEventListener("click", () => {
        window.location.href = `/member/mypage-club/${club.id}/`
    })
}

// 가입신청 버튼 클릭 시 모달창 출력
const applyBtnEvent = () => {
    document.getElementById("apply").addEventListener("click", () => {
        Swal.fire({
            title: "가입신청하시겠습니까?",
            text: `[${clubName}] 모임에 가입을 신청합니다.`,
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#3085d6",
            cancelButtonColor: "#d33",
            confirmButtonText: "신청",
            cancelButtonText: "취소",
        }).then((result) => {
            if (result.value) {
                clubDetailService.updateCMStatus(memberId, club.id)
                // 가입신청 관련 서버 작업 코드 입력
                Swal.fire("신청 완료", `[${clubName}] 모임에 가입 신청이 완료되었어요!`, "success");
            } else if (result.dismiss === "cancel") {
                return;
            }
        });
    });
}

// 승인대기 버튼 클릭 시 신청취소 모달창 출력
const cancelBtnEvent = () => {
    const cancelBtn = document.getElementById("cancel");
    cancelBtn.addEventListener("click", () => {
        Swal.fire({
            title: "가입신청을 취소하시겠습니까?",
            text: '승인대기 중입니다. 취소하시려면 "신청취소"를 눌러주세요.',
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#3085d6",
            cancelButtonColor: "#d33",
            confirmButtonText: "신청취소",
            cancelButtonText: "닫기",
        }).then((result) => {
            if (result.value) {
                clubDetailService.updateCMStatus(memberId, club.id)
                // 신청취소 관련 서버 작업 코드 입력
                Swal.fire("취소 완료", "가입 신청을 취소하였습니다.", "success");
            } else if (result.dismiss == "cancel") {
                return;
            }
        });
    });
}


// 탈퇴하기 버튼 클릭 시 탈퇴하기 모달창 출력
const quitBtnEvent = () => {
    const quitBtn = document.getElementById("quit");
    quitBtn.addEventListener("click", () => {
        Swal.fire({
            title: "모임을 탈퇴하시겠습니까?",
            text: `[${clubName}] 모임에서 탈퇴합니다.`,
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#3085d6",
            cancelButtonColor: "#d33",
            confirmButtonText: "탈퇴",
            cancelButtonText: "취소",
        }).then((result) => {
            if (result.value) {
                clubDetailService.updateCMStatus(memberId, club.id)
                // 모임탈퇴 관련 서버 작업 코드 입력
                Swal.fire("모임 탈퇴", `[${clubName}] 모임에서 탈퇴하였습니다.`, "success");
            } else if (result.dismiss == "cancel") {
                return;
            }
        });
    });
}

// 조회 결과에 따라 모임 상단 버튼을 바꿔주는 함수
const createClubTopBtn =  (clubMembers) => {
    const clubTopButtonBoxes = document.querySelector(".club-top-button-boxes")
    let clubMember = clubMembers[0]
    if (clubMembers.length === 0) {
        if (memberId === club.owner_id) {
            clubTopButtonBoxes.innerHTML = `
                <button id="manage" class="club-top-button manage">
                    <span>관리하기</span>
                </button>
            `;
            manageBtnEvent();
        } else{
            clubTopButtonBoxes.innerHTML = `
                <button id="apply" class="club-top-button apply">
                    <span>가입신청</span>
                </button>
            `;
            applyBtnEvent()
        }
    } else if (clubMember.status === -1) {
        clubTopButtonBoxes.innerHTML = `
            <button id="cancel" class="club-top-button cancel">
                <span>신청취소</span>
            </button>
        `;
        cancelBtnEvent()
    } else if (clubMember.status === 1) {
        clubTopButtonBoxes.innerHTML = `
            <button id="quit" class="club-top-button quit">
                <span>탈퇴하기</span>
            </button>
        `;
        quitBtnEvent()
    } else{
        clubTopButtonBoxes.innerHTML = `
            <button id="apply" class="club-top-button apply">
                <span>가입신청</span>
            </button>
        `;
        applyBtnEvent()
    }
}

clubDetailService.getCMInfo(memberId, club.id, createClubTopBtn);

// 상세보기 탭 부분밑 내용
const clubServiceWrap = document.querySelector("#club-service-wrap");
const activityFilterWrap = document.querySelector(".club-detail-filter-event");
const activityFilterBtn = document.querySelector(".club-detail-filter-event .club-detail-filter-button");
const infoFilterWrap = document.querySelector(".club-detail-filter-info");
const infoFilterBtn = document.querySelector(".club-detail-filter-info .club-detail-filter-button");
const tpFilterWrap = document.querySelector(".club-detail-filter-teenplay");
const tpFilterBtn = document.querySelector(".club-detail-filter-teenplay button");
const activityContent = document.querySelector(".club-detail-desc-container");
const infoContent = document.querySelector(".club-info");
const clubInfoWrap = document.querySelector(".club-info-wrap");
const tpContent = document.querySelector(".club-teenplay");
const noticeFilterWrap = document.querySelector(".club-detail-filter-notice");
const noticeFilterBtn = document.querySelector(".club-detail-filter-notice .club-detail-filter-button");
const noticeContent = document.querySelector(".club-notice");
const clubNoticeWrap = document.querySelector(".club-notice-wrap");
const clubDetailActiveWrap = document.querySelector(".club-detail-active-wrap");
const finishedEventsWrap = document.querySelector(".finished-events-wrap");
const showMoreFinishedBtnWrap = document.querySelector(".show-more-finished-btn-wrap");
const showMoreFinishedBtn = document.querySelector(".show-more-finished-btn");
const showMoreNoticeBtnWrap = document.querySelector(".show-more-notice-btn-wrap");
const showMoreNoticeBtn = document.querySelector(".show-more-notice-btn");

let page = 1
let pageNumber= 1

// 활동 클릭 시 fetch 후 뿌리는 이벤트
activityFilterBtn.addEventListener("click", () => {
    page = 1

    infoFilterWrap.style.border = "none";
    tpFilterWrap.style.border = "none";
    noticeFilterWrap.style.border = "none";

    if (!infoFilterBtn.classList.contains("off")) {
        infoFilterBtn.classList.add("off");
    }
    if (activityFilterBtn.classList.contains("off")) {
        activityFilterBtn.classList.remove("off");
    }
    if (!tpFilterBtn.classList.contains("off")) {
        tpFilterBtn.classList.add("off");
    }
    if (!noticeFilterBtn.classList.contains("off")) {
        noticeFilterBtn.classList.add("off");
    }

    activityFilterWrap.style.borderBottom = "2px solid #CE201B";

    activityContent.style.display = "block";
    infoContent.style.display = "none";
    noticeContent.style.display = "none";
    tpContent.style.display = "none";

    clubDetailService.getOAList(club.id, createListService.showOngoingList).then((text) => {
        clubDetailActiveWrap.innerHTML = text;
    })

    clubDetailService.getFAList(club.id, page, createListService.showFinishedList).then((text) => {
        finishedEventsWrap.innerHTML = text
        showMoreFAListBtnCheck()
    })
});

// 모임 정보 클릭 시 뿌리는 이벤트
infoFilterBtn.addEventListener("click", () => {
    activityFilterWrap.style.border = "none";
    tpFilterWrap.style.border = "none";
    noticeFilterWrap.style.border = "none";

    if (!activityFilterBtn.classList.contains("off")) {
        activityFilterBtn.classList.add("off");
    }
    if (infoFilterBtn.classList.contains("off")) {
        infoFilterBtn.classList.remove("off");
    }
    if (!tpFilterBtn.classList.contains("off")) {
        tpFilterBtn.classList.add("off");
    }
    if (!noticeFilterBtn.classList.contains("off")) {
        noticeFilterBtn.classList.add("off");
    }

    infoFilterWrap.style.borderBottom = "2px solid #CE201B";

    activityContent.style.display = "none";
    infoContent.style.display = "block";
    noticeContent.style.display = "none";
    tpContent.style.display = "none";

    clubInfoWrap.innerHTML = createListService.showClubInfo(club.club_info);
});

const showNoticeTap = () => {
    activityFilterWrap.style.border = "none";
    infoFilterWrap.style.border = "none";
    tpFilterWrap.style.border = "none";
    if (!activityFilterBtn.classList.contains("off")) {
        activityFilterBtn.classList.add("off");
    }
    if (!infoFilterBtn.classList.contains("off")) {
        infoFilterBtn.classList.add("off");
    }
    if (!tpFilterBtn.classList.contains("off")) {
        tpFilterBtn.classList.add("off");
    }
    if (noticeFilterBtn.classList.contains("off")) {
        noticeFilterBtn.classList.remove("off");
    }
    noticeFilterWrap.style.borderBottom = "2px solid #CE201B";

    activityContent.style.display = "none";
    infoContent.style.display = "none";
    noticeContent.style.display = "block";
    tpContent.style.display = "none";

    clubDetailService.getCNList(club.id, page, createListService.showNoticeList).then((text) => {
        clubNoticeWrap.innerHTML = text
        showMoreCNListBtnCheck()
    })
}

// 모임 공지 클릭 시 fetch 후 목록 뿌리는 이벤트
noticeFilterBtn.addEventListener("click", () => {
    page = 1
    showNoticeTap()
});

const showTeenplayTap = () => {
    activityFilterWrap.style.border = "none";
    infoFilterWrap.style.border = "none";
    noticeFilterWrap.style.border = "none";
    if (!activityFilterBtn.classList.contains("off")) {
        activityFilterBtn.classList.add("off");
    }
    if (!infoFilterBtn.classList.contains("off")) {
        infoFilterBtn.classList.add("off");
    }
    if (tpFilterBtn.classList.contains("off")) {
        tpFilterBtn.classList.remove("off");
    }
    if (!noticeFilterBtn.classList.contains("off")) {
        noticeFilterBtn.classList.add("off");
    }
    tpFilterWrap.style.borderBottom = "2px solid #CE201B";

    activityContent.style.display = "none";
    infoContent.style.display = "none";
    noticeContent.style.display = "none";
    tpContent.style.display = "block";
    clubId =club_list[0]['id']
    loadTeenplayList(clubId, pageNumber)
}

// 틴플레이 클릭 시 fetch 후 목록 뿌리는 이벤트
tpFilterBtn.addEventListener("click", () => {
    page = 1
    showTeenplayTap()
});

// 전체 모달
const modalWrap = document.querySelector(".club-modal-wrap");
// 모달 중에서 관심 설정할 때 표시할 부분
const modalLikeContainer = document.querySelector(".club-modal-like-wrap:not(.unlike)");
// 모달 중에서 관심 해제할 때 표시할 부분
const modalUnlikeContainer = document.querySelector(".club-modal-like-wrap.unlike");

//
clubDetailActiveWrap.addEventListener("click", (e) => {
    const likeWrap = e.target.closest('.club-detail-like-wrap')

    if (likeWrap) {
        const ongoingActivityId= likeWrap.querySelector('svg').classList[2];
        const ongoingLikeEmptyIcon = document.querySelector(`#empty${ongoingActivityId}`);
        const ongoingLikeFullIcon = document.querySelector(`#full${ongoingActivityId}`);

        modalWrap.style.display = "block";

        if (ongoingLikeEmptyIcon.style.display === "none") {
            clubDetailService.updateActivityLike(ongoingActivityId, false)
            modalUnlikeContainer.style.display = "block";
            modalLikeContainer.style.display = "none";
            ongoingLikeEmptyIcon.style.display = "block";
            ongoingLikeFullIcon.style.display = "none";
        } else{
            clubDetailService.updateActivityLike(ongoingActivityId, true)
            modalUnlikeContainer.style.display = "none";
            modalLikeContainer.style.display = "block";
            ongoingLikeEmptyIcon.style.display = "none";
            ongoingLikeFullIcon.style.display = "block";
        }
    }
})

//
finishedEventsWrap.addEventListener("click", (e) => {
    const likeWrap = e.target.closest('.finished-events-like-wrap')

    if (likeWrap) {
        const finishedActivityId= likeWrap.querySelector('svg').classList[2];
        const finishedLikeEmptyIcon = document.querySelector(`#empty${finishedActivityId}`);
        const finishedLikeFullIcon = document.querySelector(`#full${finishedActivityId}`);

        modalWrap.style.display = "block";

        if (finishedLikeEmptyIcon.style.display === "none") {
            clubDetailService.updateActivityLike(finishedActivityId, false)
            modalUnlikeContainer.style.display = "block";
            modalLikeContainer.style.display = "none";
            finishedLikeEmptyIcon.style.display = "block";
            finishedLikeFullIcon.style.display = "none";
        } else{
            clubDetailService.updateActivityLike(finishedActivityId, true)
            modalUnlikeContainer.style.display = "none";
            modalLikeContainer.style.display = "block";
            finishedLikeEmptyIcon.style.display = "none";
            finishedLikeFullIcon.style.display = "block";
        }
    }
})

// 모달 창 내 버튼 클릭 시 모달창 닫기
const modalLikeExitBtn = document.querySelector(".club-modal-like-button");
const modalUnlikeExitBtn = document.querySelector(".club-modal-unlike-button");

function exitModal() {
    modalWrap.style.display = "none";
}

modalLikeExitBtn.addEventListener("click", exitModal);
modalUnlikeExitBtn.addEventListener("click", exitModal);


// 공지사항 각각 제목 클릭 시 세부 내용 표시
clubNoticeWrap.addEventListener("click", (e) => {
    const clubNoticeBox = e.target.closest(".club-notice-box")

    if (clubNoticeBox) {
        const clubNoticeId= clubNoticeBox.querySelector('svg').classList[1]
        const noticeContentWraps = document.getElementById(`content-wrap${clubNoticeId}`)
        const clubNoticeShowIcon = document.getElementById(`show${clubNoticeId}`);
        const clubNoticeHiddenIcon = document.getElementById(`hidden${clubNoticeId}`);

        if (clubNoticeShowIcon.style.display === 'block') {
            clubNoticeShowIcon.style.display = 'none'
            clubNoticeHiddenIcon.style.display = 'block'
        } else {
            clubNoticeShowIcon.style.display = 'block'
            clubNoticeHiddenIcon.style.display = 'none'
        }
        if (noticeContentWraps.style.display === "none") {
            noticeContentWraps.style.display = "block"
        } else{
            noticeContentWraps.style.display = "none"
        }
    }
})


// 전달받은 date.slice(0,19)를 0월0일(0)형식으로 바꿔서 리턴하는 함수
const changeDate = (dateStr) => {
    const days = ['일', '월', '화', '수', '목', '금', '토'];
    const date = new Date(dateStr); // string타입을 date타입으로 바꿈.
    const month = date.getMonth() + 1; // getMonth()는 0부터 시작하므로 1을 더함.
    const day = date.getDate();
    const dayOfWeek = days[date.getDay()]; // 요일을 숫자로 가져와 해당 요일 문자열로 변환

    // padStart는 ,앞의 숫자의 자릿수 만큼 0을 채워줌
    return `${month.toString().padStart(2, '0')}월 ${day.toString().padStart(2, '0')}일(${dayOfWeek})`;
}

// 정보를 기준으로 목록을 만들어 주는 함수의 모듈
const createListService = (() => {
    let text = ``

    // 전달 받은 진행중인 행사 정보로 판단 후 목록이나 목록 없음 문구를 return하는 함수
    const showOngoingActivityList = (ongoingActivities) => {
        text = ``
        if (ongoingActivities.length === 0) {
            text += `
                <div class="club-detail-active-empty-wrap">
                    <div class="club-detail-active-empty-container">
                        <div class="club-detail-active-empty">진행중인 활동이 없습니다.</div>
                    </div>
                </div>
            `
        } else{
            text += `
                <div class="club-detail-active-container">
            `
            for (let ongoingActivity of ongoingActivities) {
                text += `
                    <div class="club-detail-active">
                        <div class="club-detail-img-wrap">
                            <a href="/activity/detail?id=${ongoingActivity.id}" class="club-detail-img-link">
                    `
                if (ongoingActivity.thumbnail_path) {
                    text += `
                                <img src="/upload/${ongoingActivity.thumbnail_path}" alt="활동 이미지" class="club-detail-img" />
                    `
                } else{
                    text += `
                                <img src="/static/public/web/images/logo/logo8.png" alt="활동 이미지" class="club-detail-img" />
                    `
                }
                text += `
                            </a>
                            <div class="club-detail-like-wrap">
                                <button class="club-detail-like-button">
                                    <span>
                `
                if (ongoingActivity.is_like) {
                   text += `
                                        <svg id="empty${ongoingActivity.id}" xmlns="http://www.w3.org/2000/svg" style="display: none" class="club-detail-like-icon empty ${ongoingActivity.id}" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path>
                                        </svg>
                                        <svg id="full${ongoingActivity.id} xmlns="http://www.w3.org/2000/svg" style="display: block" class="club-detail-like-icon full ${ongoingActivity.id}" viewBox="0 0 20 20" fill="currentColor">
                                            <path fill-rule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clip-rule="evenodd"></path>
                                        </svg>
                    `
                } else{
                    text += `
                                        <svg id="empty${ongoingActivity.id}" xmlns="http://www.w3.org/2000/svg" style="display: block" class="club-detail-like-icon empty ${ongoingActivity.id}" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path>
                                        </svg>
                                        <svg id="full${ongoingActivity.id}" xmlns="http://www.w3.org/2000/svg" style="display: none" class="club-detail-like-icon full ${ongoingActivity.id}" viewBox="0 0 20 20" fill="currentColor">
                                            <path fill-rule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clip-rule="evenodd"></path>
                                        </svg>
                    `
                }

                text += `
                                    </span>
                                </button>
                            </div>
                        </div>
                        <div class="club-detail-desc-box">
                            <div class="club-detail-date-wrap">
                                <div>
                                    <span class="club-detail-date"> ${changeDate(ongoingActivity.activity_start.slice(0,19))} </span>
                                </div>
                            </div>
                            <div class="event-title-wrap">
                                <a href="/activity/detail?id=${ongoingActivity.id}" class="event-title"> ${ongoingActivity.activity_title} </a>
                            </div>
                            <div class="event-detail-wrap">
                                <div class="event-usercount-wrap">
                                    <span class="event-usercount"> 참여 ${ongoingActivity.participant_count} </span>
                                </div>
                            </div>
                        </div>
                    </div>
                `
            }
            text += `    
                </div>
            `
        }
        return text;
    }

    // 전달 받은 종료된 행사 정보로 판단 후 목록이나 목록 없음 문구를 return하는 함수
    const showFinishedActivityList = (finishedActivities) => {
        text = ``

        if (finishedActivities.length === 0) {
            text += `
                <div class="club-finished-events-empty-wrap">
                    <div class="club-finished-events-empty-container">
                        <div class="club-finished-events-empty">종료된 활동이 없습니다.</div>
                    </div>
                </div>
            `
        } else{
            text += `
                <div class="finished-events-container">
            `
            for (let finishedActivity of finishedActivities){
                text += `
                    <div class="finished-events-boxes">
                        <div class="finished-events-img-wrap">
                            <a href="/activity/detail?id=${finishedActivity.id}" class="finished-events-img-link">
                `
                if (finishedActivity.thumbnail_path) {
                    text += `
                                <img src="/upload/${finishedActivity.thumbnail_path}" alt="활동 이미지" class="finished-events-img" />
                `
                } else{
                    text += `
                                <img src="/teenplay_server/static/public/web/images/logo/logo8.png" alt="활동 이미지" class="club-detail-img" />
                `
                }
                text += `
                                <div class="finished-events-img-back"></div>
                            </a>
                            <div class="finished-events-like-wrap">
                                <button class="finished-events-like-btn">
                                    <span>
                `
                if (finishedActivity.is_like) {
                   text += `
                                        <svg id="empty${finishedActivity.id}" xmlns="http://www.w3.org/2000/svg" style="display: none" class="club-detail-like-icon empty ${finishedActivity.id}" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path>
                                        </svg>
                                        <svg id="full${finishedActivity.id}" xmlns="http://www.w3.org/2000/svg" style="display: block" class="club-detail-like-icon full ${finishedActivity.id}" viewBox="0 0 20 20" fill="currentColor">
                                            <path fill-rule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clip-rule="evenodd"></path>
                                        </svg>
                    `
                } else{
                    text += `
                                        <svg id="empty${finishedActivity.id}" xmlns="http://www.w3.org/2000/svg" style="display: block" class="club-detail-like-icon empty ${finishedActivity.id}" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path>
                                        </svg>
                                        <svg id="full${finishedActivity.id}" xmlns="http://www.w3.org/2000/svg" style="display: none" class="club-detail-like-icon full ${finishedActivity.id}" viewBox="0 0 20 20" fill="currentColor">
                                            <path fill-rule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clip-rule="evenodd"></path>
                                        </svg>
                    `
                }

                text += `
                                    </span>
                                </button>
                            </div>
                        </div>
                        <div class="finished-events-desc-wrap">
                            <div class="finished-events-finish-wrap">
                                <div>
                                    <div class="finished-events-finish">종료</div>
                                </div>
                            </div>
                            <div class="finished-events-name-wrap">
                                <a href="/activity/detail?id=${finishedActivity.id}" class="finished-events-name"> ${finishedActivity.activity_title} </a>
                            </div>
                            <div class="finished-events-price-wrap">
                                <div class="finished-events-count-wrap">
                                    <span class="finished-events-count"> 참여 ${finishedActivity.participant_count} </span>
                                </div>
                            </div>
                        </div>
                    </div>
                `
            }
            text += `
                </div>
            `
        }
        return text;
    }

    // 전달 받은 모임 정보로 판단 후 모임 정보나 정보 없음 문구를 return하는 함수
    const showClubInfo = (clubInfo) => {
        text = ``

        if (!clubInfo) {
            text += `
                <div class="club-info-empty-wrap">
                    <div class="club-info-empty-container">
                        <div class="club-info-empty">등록된 모임 정보가 없습니다.</div>
                    </div>
                </div>
            `
        } else{
            text += `
                <div class="club-info-container">
                    <div class="club-info-texts">${ clubInfo.replace(/\n/g, '<br>') }</div>
                </div>
            `
        }
        return text;
    }

    // 전달 받은 모임 공지사항 정보로 판단 후 목록이나 목록 없음 문구를 return하는 함수
    const showClubNoticeList = (clubNotices) => {
        text = ``

        if (clubNotices.length === 0) {
            text += `
                <div class="club-notice-empty-wrap">
                    <div class="club-notice-empty-container">
                        <div class="club-notice-empty">등록된 공지사항이 없습니다.</div>
                    </div>
                </div>
            `
        } else{
            text += `
                <div class="club-notice-container">
            `
            for (let clubNotice of clubNotices)
            text += `
                    <div class="club-notice-boxes-border">
                        <div class="club-notice-boxes">
                            <div class="club-notice-box ${clubNotice.id}">
                                <div>
                                    <div class="club-notice-title-wrap">
                                        <div class="club-notice-title-container">
                                            <!-- 이 안에 제목이 들어갑니다. -->
                                            <div class="club-notice-title">${clubNotice.notice_title}</div>
                                        </div>
                                        <svg id="show${clubNotice.id}" xmlns="http://www.w3.org/2000/svg" style="display: block" class="club-notice-show-icon ${clubNotice.id}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                            <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7"></path>
                                        </svg>
                                        <svg id="hidden${clubNotice.id}" xmlns="http://www.w3.org/2000/svg" style="display: none" class="club-notice-hide-icon ${clubNotice.id}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                            <path stroke-linecap="round" stroke-linejoin="round" d="M5 15l7-7 7 7"></path>
                                        </svg>
                                    </div>
                                    <div class="club-notice-created-date">${clubNotice.created_date.slice(0, 10)}</div>
                                </div>
                            </div>
                            <div id="content-wrap${clubNotice.id}" class="club-notice-content-wrap ${clubNotice.id}" style="display: none">
                                <!-- 이 안에 내용이 들어갑니다. -->
                                <div class="club-notice-content">${clubNotice.notice_content.replace(/\n/g, '<br>')}</div>
                            </div>
                        </div>
                    </div>
            `
            text += `
                </div>
            `
        }
        return text;
    }

    // 전달 받은 틴플레이 정보로 판단 후 목록이나 목록 없음 문구를 return하는 함수
    const showTeenplayList = () => {
        text = ``
        return text;
    }

    return {
        showOngoingList: showOngoingActivityList,
        showFinishedList: showFinishedActivityList,
        showClubInfo: showClubInfo,
        showNoticeList: showClubNoticeList,
        showTeenplayList: showTeenplayList
    }
})()

// fetch를 통해 종료된 행사의 다음페이지가 있는지 검사하고 더보고 버튼을 보이게할지 변경하는 함수
const showMoreFAListBtnCheck = () => {
    clubDetailService.getFAList(club.id, page + 1).then((finishedActivities) => {
        if (finishedActivities.length !== 0) {
            showMoreFinishedBtnWrap.style.display = "block";
        } else {
            showMoreFinishedBtnWrap.style.display = "none";
        }
    })
}

// 더보기 클릭 시 fetch를 통해 종료된 행사 목록의 다음 페이지 정보를 가져와 추가해서 보여주는 이벤트
showMoreFinishedBtn.addEventListener("click", () => {
    clubDetailService.getFAList(club.id, ++page, createListService.showFinishedList).then((text) => {
        finishedEventsWrap.innerHTML += text
    })
    // likeEvent()
    showMoreFAListBtnCheck()
})

// fetch를 통해 모임 공지사항의 다음페이지가 있는지 검사하고 더보고 버튼을 보이게할지 변경하는 함수
const showMoreCNListBtnCheck = () => {
    clubDetailService.getCNList(club.id, page + 1).then((clubNotices) => {
        if (clubNotices.length !== 0) {
            showMoreNoticeBtnWrap.style.display = "block";
        } else {
            showMoreNoticeBtnWrap.style.display = "none";
        }
    })
}

// 더보기 클릭 시 fetch를 통해 모임 공지사항 목록의 다음 페이지 정보를 가져와 추가해서 보여주는 이벤트
showMoreNoticeBtn.addEventListener("click", () => {
    clubDetailService.getCNList(club.id, ++page, createListService.showNoticeList).then((text) => {
        clubNoticeWrap.innerHTML += text
    })
    showMoreCNListBtnCheck()
})

if (club.view === 'notice') {
    showNoticeTap()
} else if (club.view === 'activity') {
    // 페이지 로드 시 fetch를 통해 진행중인 행사 정보를 가져와 넣어주는 이벤트
    clubDetailService.getOAList(club.id, createListService.showOngoingList).then((text) => {
        clubDetailActiveWrap.innerHTML = text;
    })

    // 페이지 로드 시 fetch를 통해 종료된 행사 정보를 가져와 넣어주는 이벤트
    clubDetailService.getFAList(club.id, page, createListService.showFinishedList).then((text) => {
        finishedEventsWrap.innerHTML = text
        showMoreFAListBtnCheck()
    })
} else{
    showTeenplayTap()
}




// 틴플레이 관련 양현이 작성한 코드

// 틴플레이 업로드 버튼 클릭 시 모달창
const tpUploadModal = document.querySelector(".club-upload-modal-wrap");
const tpUploadBtn = document.querySelector(".club-detail-filter-upload-wrap");
const tpModalCloseBtn = document.querySelector(".upload-modal-close-wrap");
tpUploadBtn.addEventListener("click", () => {
    tpUploadModal.style.display = "block";
});
tpModalCloseBtn.addEventListener("click", () => {
    tpUploadModal.style.display = "none";
});

// 파일 선택 클릭 시 파일 첨부가능하도록
const fileInput = document.querySelector("input[name=Filedata]");
const addButton = document.querySelector(".upload-modal-upload-button-wrap");

addButton.addEventListener("click", () => {
    fileInput.click();
});

// 아이콘 클릭 시 파일 첨부가능하도록
const uploadIcon = document.querySelector(".upload-modal-content-icon-wrap");
uploadIcon.addEventListener("click", () => {
    fileInput.click();
});

// 파일 첨부 시 용량(20MB 이하) 체크하여 클 시 에러메시지 출력 및
// 용량 문제 없을 시 파일 정보를 아래에 표시
const modalContentMsg = document.querySelector(".upload-modal-content-msg");
const fileSizeInfo = document.querySelector(".pr-file-size");
const fileNameInfo = document.querySelector(".pr-file-name");
const afterUploadModal = document.querySelector(".upload-modal-content-container.after");
const beforeUploadModal = document.querySelector(".upload-modal-content-container.before");
const fileRemoveBtn = document.querySelector(".pr-remove-btn");

function checkFileSize(obj, size) {
    let check = false;
    let sizeInBytes = obj.files[0].size;
    if (sizeInBytes > size) {
        check = false;
    } else {
        check = true;
    }
    return check;
}

function getFileSizeWithExtension(sizeInBytes) {
    let fileSizeExt = new Array("bytes", "kb", "mb", "gb");
    let i = 0;
    let checkSize = sizeInBytes;
    while (checkSize > 900) {
        checkSize /= 1024;
        i++;
    }
    checkSize = Math.round(checkSize * 100) / 100 + "" + fileSizeExt[i];
    return checkSize;
}

const MAX_SIZE = 20; // 20MB
fileInput.addEventListener("change", (e) => {
    if (e.target.value) {
        let checkSize = 1024 * 1024 * MAX_SIZE;
        if (!checkFileSize(fileInput, checkSize)) {
            modalContentMsg.style.color = "#CE201B";
            e.preventDefault();
            return;
        }
        modalContentMsg.style.color = "#606060";
        fileSize = e.target.files[0].size;
        fileSizeInfo.innerText = getFileSizeWithExtension(fileSize);
        fileNameInfo.innerText = e.target.files[0].name;
        beforeUploadModal.classList.remove("appear");
        beforeUploadModal.classList.add("disappear");
        setTimeout(() => {
            beforeUploadModal.classList.add("hidden");
            afterUploadModal.classList.remove("hidden");
            afterUploadModal.classList.remove("disappear");
            afterUploadModal.classList.add("appear");
        }, 501);
    }
});

// 파일 첨부 후 x버튼으로 삭제하기
fileRemoveBtn.addEventListener("click", (e) => {
    e.preventDefault();
    fileInput.value = "";
    afterUploadModal.classList.remove("appear");
    afterUploadModal.classList.add("disappear");
    setTimeout(() => {
        afterUploadModal.classList.add("hidden");
        beforeUploadModal.classList.remove("hidden");
        beforeUploadModal.classList.remove("disappear");
        beforeUploadModal.classList.add("appear");
    }, 501);
    fileSizeInfo.innerText = "";
    fileNameInfo.innerText = "";
});

// 드래그 앤 드롭으로 파일 첨부하기
beforeUploadModal.addEventListener("dragenter", (e) => {
    e.preventDefault();
});
beforeUploadModal.addEventListener("dragover", (e) => {
    e.preventDefault();
});
beforeUploadModal.addEventListener("dragleave", (e) => {
    e.preventDefault();
});
beforeUploadModal.addEventListener("drop", (e) => {
    e.preventDefault();
    let file = e.dataTransfer;
    if (!checkFileSize(file, 1024 * 1024 * MAX_SIZE)) {
        modalContentMsg.style.color = "#CE201B";
        return;
    }
    modalContentMsg.style.color = "#606060";
    fileSize = file.files[0].size;
    fileSizeInfo.innerText = getFileSizeWithExtension(fileSize);
    fileNameInfo.innerText = file.files[0].name;
    beforeUploadModal.classList.remove("appear");
    beforeUploadModal.classList.add("disappear");
    setTimeout(() => {
        beforeUploadModal.classList.add("hidden");
        afterUploadModal.classList.remove("hidden");
        afterUploadModal.classList.remove("disappear");
        afterUploadModal.classList.add("appear");
    }, 501);
});

// 썸네일 업로드 창 클릭 시 파일 첨부 가능하도록
const thumbnailInput = document.querySelector("#background-image");
const thumbnailUploadBox = document.querySelector(".cover-thumbnail-wrap");

thumbnailUploadBox.addEventListener("click", () => {
    thumbnailInput.click();
});

// 썸네일 용량 제한(10MB) 초과 시 에러메시지 출력 / 괜찮을 시 정보 보이게
const thumbnailSizeMsg = document.querySelector(".img-form-profile-size-error");
const THUMBNAIL_SIZE = 10;
const thumbnailSizeInfo = document.querySelector(".pr-thumbnail-size");
const thumbnailNameInfo = document.querySelector(".pr-thumbnail-name");
const uploadedThumbnailInfo = document.querySelector(".pr-write-uploaded-thumbnail-container");

thumbnailInput.addEventListener("change", (e) => {
    if (e.target.value) {
        let checkSize = 1024 * 1024 * THUMBNAIL_SIZE;
        if (!checkFileSize(thumbnailInput, checkSize)) {
            thumbnailSizeMsg.style.display = "block";
            e.preventDefault();
            return;
        }
        thumbnailSizeMsg.style.display = "none";
        fileSize = e.target.files[0].size;
        thumbnailSizeInfo.innerText = getFileSizeWithExtension(fileSize);
        thumbnailNameInfo.innerText = e.target.files[0].name;
        thumbnailUploadBox.classList.remove("appear");
        thumbnailUploadBox.classList.add("disappear");
        setTimeout(() => {
            thumbnailUploadBox.classList.add("hidden");
            uploadedThumbnailInfo.classList.remove("hidden");
            uploadedThumbnailInfo.classList.remove("disappear");
            uploadedThumbnailInfo.classList.add("appear");
        }, 501);
    }
});

// 드래그 앤 드롭으로 썸네일 첨부하기
// thumbnailUploadBox.addEventListener("dragenter", (e) => {
//     e.preventDefault();
// });
// thumbnailUploadBox.addEventListener("dragover", (e) => {
//     e.preventDefault();
// });
// thumbnailUploadBox.addEventListener("dragleave", (e) => {
//     e.preventDefault();
// });
// thumbnailUploadBox.addEventListener("drop", (e) => {
//     e.preventDefault();
//     let file = e.dataTransfer;
//     let checkSize = 1024 * 1024 * THUMBNAIL_SIZE;
//     if (!checkFileSize(file, checkSize)) {
//         thumbnailSizeMsg.style.display = "block";
//         e.preventDefault();
//         return;
//     }
//     thumbnailSizeMsg.style.display = "none";
//     fileSize = file.files[0].size;
//     thumbnailSizeInfo.innerText = getFileSizeWithExtension(fileSize);
//     thumbnailNameInfo.innerText = file.files[0].name;
//     thumbnailUploadBox.classList.remove("appear");
//     thumbnailUploadBox.classList.add("disappear");
//     setTimeout(() => {
//         thumbnailUploadBox.classList.add("hidden");
//         uploadedThumbnailInfo.classList.remove("hidden");
//         uploadedThumbnailInfo.classList.remove("disappear");
//         uploadedThumbnailInfo.classList.add("appear");
//     }, 501);
// });

// 썸네일 첨부 후 x버튼으로 삭제하기
const thumbnailRemoveBtn = document.querySelector(".pr-thumbnail-remove-btn");
const teenPlayTextInput = document.querySelector(".name-form-input");
const finalSaveButton = document.querySelector(".upload-modal-confirm-button-wrap");

thumbnailRemoveBtn.addEventListener("click", (e) => {
    e.preventDefault();
    thumbnailInput.value = "";
    uploadedThumbnailInfo.classList.remove("appear");
    uploadedThumbnailInfo.classList.add("disappear");
    setTimeout(() => {
        uploadedThumbnailInfo.classList.add("hidden");
        thumbnailUploadBox.classList.remove("hidden");
        thumbnailUploadBox.classList.remove("disappear");
        thumbnailUploadBox.classList.add("appear");
    }, 501);
    thumbnailSizeInfo.innerText = "";
    thumbnailNameInfo.innerText = "";
    if (!finalSaveButton.classList.contains("disabled")) {
        finalSaveButton.classList.add("disabled");
    }
});

// 틴플레이 설명 및 썸네일 업로드 시 업로드 버튼 활성화 (10자 이상)
teenPlayTextInput.addEventListener("keyup", () => {
    if (teenPlayTextInput.value.length >= 10 && thumbnailInput.files.length) {
        finalSaveButton.classList.remove("disabled");
    } else {
        if (!finalSaveButton.classList.contains("disabled")) {
            finalSaveButton.classList.add("disabled");
        }
    }
});

thumbnailInput.addEventListener("change", (e) => {
    if (e.target.files.length && teenPlayTextInput.value.length >= 10) {
        finalSaveButton.classList.remove("disabled");
    } else {
        if (!finalSaveButton.classList.contains("disabled")) {
            finalSaveButton.classList.add("disabled");
        }
    }
});

// 틴플레이 클럽 버튼 클릭 시 최초 업로드 함수
function loadTeenplayList(clubId, page){
    clubDetailService.getTeenplayList(clubId, page).then((teenplayInfo) => {
        let teenplayFirstInfo =teenplayInfo.teenplay_list
        // let containTeenplay = teenplayWrap.getElementsByClassName('club-teenplay-contents').length

        if (teenplayFirstInfo.length > 0){
            teenplayWrap.innerHTML = ''
            teenplayWrap.innerHTML += showList(teenplayFirstInfo)
            if(teenplayInfo.has_next){
                teenplayMoreButtonWrap.style.display='flex'
            }
            else{
                teenplayMoreButtonWrap.style.display = "none"
            }
        }
        else{
            teenplayMoreButtonWrap.style.display = "none"
            teenplayViewList.innerHTML = noneText
        }
    })
}
