NodeList.prototype.filter = Array.prototype.filter;
NodeList.prototype.map = Array.prototype.map;

// // 관심 활동 클릭 시 이미지 변경 및 모달창 띄우기
let heartCheck = document.querySelector(".like-act-button");
// 관심 행사 클릭 시 모달창 띄우고 삭제하기 위해 확인
let likeModal = document.querySelector(".k-club-modal-wrap");

let likeTextIcon = document.querySelector(".k-club-modal-like-title-text");
let likeTextIconDelete = document.querySelector(".k-club-modal-like-title-text-delete");


// 관심 활동 추가/삭제 모듈
const addActivityLike = async (activityId, isCreate) => {
    await fetch(`/activity/like?id=${activityId}&is-create=${isCreate}`);
}


// 하트 이미지 색상 변경 함수
const heartColor = document.querySelector(".like-act-button-image");

// 관심 활동 모달창 출력/숨김
const addLikeModal = () => {
    likeModal.style.display = "block";
    likeTextIcon.style.display = "flex";
    likeTextIconDelete.style.display = "none";
}
const cancelLikeModal = () => {
    likeModal.style.display = "block";
    likeTextIcon.style.display = "none";
    likeTextIconDelete.style.display = "flex";
}
const changeHeartColor = () => {
    let color = window.getComputedStyle(heartColor);
    // 하트가 비어있다면
    if (heartColor && color.getPropertyValue("fill") === "none") {
        heartColor.style.fill = "#CE201B";
        return true;
    }
    heartColor.style.fill = "none";
    return false;
}

// 이미 관심활동이라면 하트 채워놓기
const isLike = document.querySelector("input[name=is-like]").value;
if (isLike === 'True') {
    changeHeartColor();
}

// 클릭 시 관심활동 등록하기 및 관심활동 등록 회원 수 업데이트
const activityId = document.querySelector("input[name=activity-id]").value;
const activityLikeCount = document.querySelector(".activity-like-count");
const showLikeCount = (likeCount) => {
    activityLikeCount.innerHTML = `관심 ${likeCount}명 | `;
}
// 업데이트하고 시작
activityLikeCountService.getCount(activityId, showLikeCount);

// 클릭 시에도 작동하도록 등록
heartCheck.addEventListener("click", async (e) => {
    let check = changeHeartColor();
    if (check) {
        await addActivityLike(activityId, true);
        await activityLikeCountService.getCount(activityId, showLikeCount);
        addLikeModal();
    }
    else {
        await addActivityLike(activityId, false);
        await activityLikeCountService.getCount(activityId, showLikeCount);
        cancelLikeModal();
    }
});


// 관심 활동 선택 후 닫기 버튼을 클릭했을 때 모달창 닫기
let closeLikeModal = document.querySelector(".k-club-modal-like-button");
closeLikeModal.addEventListener("click", (e) => {
    likeModal.style.display = "none";
});

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// // 특정 버튼 클릭 시 해당 위치로 이동
document.querySelector(".act-intro").addEventListener("click", () => {
    // 이동하려는 대상 div 선택
    const targetDiv = document.querySelector(".summary-box-one");

    window.scrollTo({
        top: targetDiv.offsetTop,
        behavior: "smooth",
    });
});

document.querySelector(".act-location").addEventListener("click", () => {
    const targetDiv = document.querySelector(".flex-items-end-container");

    window.scrollTo({
        top: targetDiv.offsetTop,
        behavior: "smooth",
    });
});

document.querySelector(".act-info").addEventListener("click", () => {
    const targetDiv = document.querySelector(".club-section");

    window.scrollTo({
        top: targetDiv.offsetTop,
        behavior: "smooth",
    });
});

document.querySelector(".act-noti").addEventListener("click", () => {
    const targetDiv = document.querySelector(".feed-main-title-box");

    window.scrollTo({
        top: targetDiv.offsetTop,
        behavior: "smooth",
    });
});

document.querySelector(".act-inquiry").addEventListener("click", () => {
    const targetDiv = document.querySelector(".comment-title");

    window.scrollTo({
        top: targetDiv.offsetTop,
        behavior: "smooth",
    });
});

document.querySelector(".act-suggestion").addEventListener("click", () => {
    const targetDiv = document.querySelector(".activity-list-more-title");

    window.scrollTo({
        top: targetDiv.offsetTop - 30,
        behavior: "smooth",
    });
});

// document.querySelector(".act-cancel").addEventListener("click", () => {
//     const targetDiv = document.querySelector(".policy-title");
//
//     window.scrollTo({
//         top: targetDiv.offsetTop,
//         behavior: "smooth",
//     });
// });
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// // 특정 위치로 이동되었을 때 act 아래쪽에 줄이 생기는 액션
// 버튼 선택 시 스타일 변경을 위한 설정  (버튼의 클래스에 따라 변경)
let introLine = document.querySelector(".act-intro");
let locationLine = document.querySelector(".act-location");
let infoLine = document.querySelector(".act-info");
let notiLine = document.querySelector(".act-noti");
let inquiryLine = document.querySelector(".act-inquiry");
let suggetsionLine = document.querySelector(".act-suggestion");

// 특정 div의 위치를 설정
let targetEndContainerDiv = document.querySelector(".flex-items-end-container");
let targetMapDiv = document.querySelector(".map-text-title");
let targetFeedDiv = document.querySelector(".feed-main-title-box");
let targetCommentDiv = document.querySelector(".comment-title");
let targetActivityMore = document.querySelector(".activity-list-more-title");

window.addEventListener("scroll", function () {
    let targetEndContainerPosition = targetEndContainerDiv.offsetTop;
    let targetMapPosition = targetMapDiv.offsetTop;
    let targetFeedPosition = targetFeedDiv.offsetTop;
    let targetCommentPosition = targetCommentDiv.offsetTop;
    let targetActivityMorePosition = targetActivityMore.offsetTop;

    if (document.documentElement.scrollTop < targetEndContainerPosition) {
        // 스크롤 위치에 따라 적절히 조정
        introLine.style.borderBottomWidth = "2px";
        introLine.style.color = "#ce201b";
        locationLine.style.color = "";
        locationLine.style.borderColor = "";
        infoLine.style.color = "";
        infoLine.style.borderColor = "";
        notiLine.style.color = "";
        notiLine.style.borderColor = "";
        inquiryLine.style.color = "";
        inquiryLine.style.borderColor = "";
        suggetsionLine.style.color = "";
        suggetsionLine.style.borderColor = "";
    }
    // 활동 장소
    if (document.documentElement.scrollTop >= targetEndContainerPosition && document.documentElement.scrollTop < targetMapPosition) {
        locationLine.style.color = "#ce201b";
        locationLine.style.borderColor = "#ce201b";
        introLine.style.borderBottomWidth = "0";
        introLine.style.color = "rgb(135 141 145 / var(--tw-text-opacity))";
        introLine.style.transition = "color 0.2s, border-bottom-color 0.2s";
        infoLine.style.color = "";
        infoLine.style.borderColor = "";
        notiLine.style.color = "";
        notiLine.style.borderColor = "";
        inquiryLine.style.color = "";
        inquiryLine.style.borderColor = "";
        suggetsionLine.style.color = "";
        suggetsionLine.style.borderColor = "";
    }
    // 활동 정보
    if (document.documentElement.scrollTop >= targetMapPosition && document.documentElement.scrollTop < targetFeedPosition) {
        locationLine.style.color = "";
        locationLine.style.borderColor = "";
        notiLine.style.color = "";
        notiLine.style.borderColor = "";
        inquiryLine.style.color = "";
        inquiryLine.style.borderColor = "";
        suggetsionLine.style.color = "";
        suggetsionLine.style.borderColor = "";
        infoLine.style.color = "#ce201b";
        infoLine.style.borderColor = "#ce201b";
        introLine.style.transition = "color 0.3s, border-bottom-color 0.3s";
    }
    // 모임 공지
    if (document.documentElement.scrollTop >= targetFeedPosition && document.documentElement.scrollTop < targetCommentPosition) {
        locationLine.style.color = "";
        locationLine.style.borderColor = "";
        infoLine.style.color = "";
        infoLine.style.borderColor = "";
        inquiryLine.style.color = "";
        inquiryLine.style.borderColor = "";
        suggetsionLine.style.color = "";
        suggetsionLine.style.borderColor = "";
        notiLine.style.color = "#ce201b";
        notiLine.style.borderColor = "#ce201b";
        notiLine.style.transition = "color 0.3s, border-bottom-color 0.3s";
    }
    // 댓글
    if (document.documentElement.scrollTop >= targetCommentPosition && document.documentElement.scrollTop < targetActivityMorePosition) {
        locationLine.style.color = "";
        locationLine.style.borderColor = "";
        infoLine.style.color = "";
        infoLine.style.borderColor = "";
        notiLine.style.color = "";
        notiLine.style.borderColor = "";
        suggetsionLine.style.color = "";
        suggetsionLine.style.borderColor = "";
        inquiryLine.style.color = "#ce201b";
        inquiryLine.style.borderColor = "#ce201b";
        notiLine.style.transition = "color 0.3s, border-bottom-color 0.3s";
    }
    // 추천
    if (document.documentElement.scrollTop > targetActivityMorePosition) {
        locationLine.style.color = "";
        locationLine.style.borderColor = "";
        infoLine.style.color = "";
        infoLine.style.borderColor = "";
        notiLine.style.color = "";
        notiLine.style.borderColor = "";
        inquiryLine.style.color = "";
        inquiryLine.style.borderColor = "";
        suggetsionLine.style.color = "#ce201b";
        suggetsionLine.style.borderColor = "#ce201b";
        suggetsionLine.style.transition = "color 0.1s, border-bottom-color 0.1s";
    }
});
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


// // 접기 버튼 클릭 시 동작
// 숨겨진 이미지 확인 및 버튼 모양 변경
filpClickBtn = document.querySelector(".filp-button");
filpHiddenClickBtn = document.querySelector(".filp-button-hidden");
filpMoreClickBtn = document.querySelector(".filp-more-hidden");
overflowHidden = document.querySelector(".max-overflow-hidden");
isHiddenShadow = document.querySelector(".is-hidden-shadow");

// 활동 내용의 높이에 따라 활동소개더보기 버튼 표시 유무 및 아랫쪽 흐려짐 여부 설정
const activityContentBox = document.querySelector(".overflow-hidden-box");
console.log(activityContentBox.offsetHeight);
const flipButtonContainer = document.querySelector(".filp-more");
if (activityContentBox.offsetHeight <= window.innerHeight * 0.5) {
    flipButtonContainer.style.display = "none";
    isHiddenShadow.style.backgroundImage = "none";
} else {
    flipButtonContainer.style.display = "block";
    isHiddenShadow.style.backgroundImage = "linear-gradient(to top, var(--tw-gradient-stops))";
}
filpClickBtn.addEventListener("click", (e) => {
    overflowHidden.style.maxHeight = "none";
    isHiddenShadow.style.backgroundImage = "none";
    filpClickBtn.style.display = "none";
    filpHiddenClickBtn.style.display = "flex";
});

filpHiddenClickBtn.addEventListener("click", () => {
    overflowHidden.style.maxHeight = "700px";
    isHiddenShadow.style.backgroundImage = "linear-gradient(to top, var(--tw-gradient-stops))";
    filpClickBtn.style.display = "flex";
    filpHiddenClickBtn.style.display = "none";
});


// 관심행사 추가 시 발생되는 동작
let nonLikeDisplay = document.querySelectorAll(".k-like-btn-shadow");
let nonLikeBtn = document.querySelectorAll(".k-like-btn-display");
let likeBtn = document.querySelectorAll(".k-like-btn-display-none");
let subLikeModal = document.querySelector(".k-club-modal-wrap");
let subLikeTextIcon = document.querySelector(".k-club-modal-like-title-text");
let subLikeTextIconDelete = document.querySelector(".k-club-modal-like-title-text-delete");

const isLikeds = document.querySelectorAll("input[name=is-liked]")
nonLikeDisplay.forEach(async (displayButton, i) => {
    let isLiked = isLikeds[i].value;
    if (isLiked === 'True') {
        nonLikeBtn[i].style.display = "none";
        likeBtn[i].style.display = "flex";
    } else {
        nonLikeBtn[i].style.display = "flex";
        likeBtn[i].style.display = "none";
    }
    await displayButton.addEventListener("click", (e) => {
        nonLikeBtn.forEach(async (nonLikeButton, j) => {
            if (i === j) {
                let recommendedActivityId = document.querySelectorAll("input[name=recommended-activity-id]")[i].value
                if (nonLikeBtn[j].style.display === "none") {
                    await addActivityLike(recommendedActivityId, false);
                    nonLikeBtn[j].style.display = "flex";
                    likeBtn[j].style.display = "none";
                    subLikeModal.style.display = "flex";
                    subLikeTextIcon.style.display = "none";
                    subLikeTextIconDelete.style.display = "flex";

                } else {
                    await addActivityLike(recommendedActivityId, true);
                    nonLikeBtn[j].style.display = "none";
                    likeBtn[j].style.display = "flex";
                    subLikeModal.style.display = "flex";
                    subLikeTextIconDelete.style.display = "none";
                    subLikeTextIcon.style.display = "flex";
                }
            }
        });
    });
});

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// 쪽지 보내기 클릭 시 쪽지 보내기 모달 출력 이벤트
const sendLetterBoxBtn = document.querySelector(".send-letter-btn");
const sendLetter = document.querySelector(".send-modal-wrap");
const senderInfo = document.querySelector(".send-sender-email")
const receiverInfo = document.querySelector(".send-receiver-email")

// 쪽지 모달창에 발신자 정보 받아와서 넣기
const sendLetterAddInfo = (replyId) => {
    const memberName = document.querySelector("input[name=member-name]").value;
    const memberEmail = document.querySelector("input[name=member-email]").value;
    senderInfo.innerText = `${memberName} (${memberEmail})`;
    const receiverName = document.querySelector(`.member-name${replyId}`).innerText;
    const receiverEmail = document.querySelector(`.member-email${replyId}`).value;
    receiverInfo.innerText = `${receiverName} (${receiverEmail})`;
}

// 틴친 여부에 따라 버튼 다르게 표시하기
/////////////////////////////////////
// 틴친 신청 버튼
const teenchinAddButton = document.querySelector(".teenchin-add-btn");
// 틴친 신청취소 버튼
const teenchinCancelButton = document.querySelector(".teenchin-request-btn");
// 틴친 수락/거절 버튼
const teenchinAcceptButton = document.querySelector(".teenchin-accept-btn");
// 틴친 끊기 버튼
const teenchinDeleteButton = document.querySelector(".teenchin-btn");

// 보여주는 함수
const helpShowButton = (button) => {
    if (button.classList.contains("hidden")) {
        button.classList.remove("hidden");
    }
}
// 숨겨주는 함수
const helpHideButton = (button) => {
    if (!button.classList.contains("hidden")) {
        button.classList.add("hidden");
    }
}

// 이제 위 요소들을 사용하여 틴친 상태에 따라 버튼을 바꿔줄 함수 정의
const showButtonsByTeenchinStatus = (teenchinStatus) => {
    let status = teenchinStatus.teenchinStatus;
    let isSender = teenchinStatus.isSender;
    if (status === 0) {
        helpShowButton(teenchinAddButton);
        helpHideButton(teenchinCancelButton);
        helpHideButton(teenchinAcceptButton)
        helpHideButton(teenchinDeleteButton);
    } else if (status === 1) {
        helpShowButton(teenchinDeleteButton);
        helpHideButton(teenchinAddButton);
        helpHideButton(teenchinAcceptButton)
        helpHideButton(teenchinCancelButton);
    } else if (isSender) {
        helpShowButton(teenchinCancelButton);
        helpHideButton(teenchinAcceptButton);
        helpHideButton(teenchinAddButton);
        helpHideButton(teenchinDeleteButton);
    } else {
        helpShowButton(teenchinAcceptButton);
        helpHideButton(teenchinAddButton);
        helpHideButton(teenchinCancelButton);
        helpHideButton(teenchinDeleteButton);
    }
}

// 위에서 정의한 함수를 사용할 때, 댓글에서 프로필 사진을 클릭하면 
// 해당 멤버의 id를 같이 넘겨 틴친 상태에 따라 버튼을 바로 바꿔줘야 합니다.
// 따라서 프로필 모달이 표시됨과 동시에 이루어지도록 합니다.

// 틴친 클릭 시 프로필 모달 나오도록 하기
const profileModal = document.querySelector("div.profile");
const profileModalProfileImage = document.querySelector(".profile-default-img");
const profileModalMemberName = document.querySelector("div.profile-name");
let opponentTeenchinId = 0;
const showMemberProfileModal = async (replyId) => {
    opponentTeenchinId = document.querySelector(`.reply-writer-id${replyId}`).value;
    if (memberId === opponentTeenchinId) return;
    if (profileModal.classList.contains("hidden")) {
        profileModal.classList.remove("hidden")
        const memberProfileImage = document.querySelector(`.profile-image${replyId}`);
        profileModalProfileImage.src = memberProfileImage.src;
        const memberProfileName = document.querySelector(`.member-name${replyId}`);
        profileModalMemberName.innerText = memberProfileName.innerText;
        sendLetterAddInfo(replyId);
        await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
    }
}

const hideMemberProfileModal = () => {
    if (!profileModal.classList.contains("hidden")){
        profileModal.classList.add("hidden");
    }
}

const addClickEventReplyProfile = () => {
    const profilePhotos = document.querySelectorAll(".k-comment-profile-container");
    profilePhotos.forEach((wrap) => {
        wrap.addEventListener("click", (e) => {
            let replyId = wrap.classList[1];
            showMemberProfileModal(replyId);
        })
    })
    const modalDivision = document.querySelector(".modal-divison")
    const modalContainer = document.querySelector(".teenchin-box.post-update-wrap")
    modalDivision.addEventListener("click", (e) => {
        if (e.target !== modalContainer){
            hideMemberProfileModal()
        }
    })
}

// 프로필 클릭 시 틴친 프로필 모달 출력 이벤트
const commentProfileImg = document.querySelector(".k-comment-profile-container");
const profile = document.querySelector(".profile");

if (commentProfileImg){
    commentProfileImg.addEventListener("click", () => {
        profile.classList.remove("hidden");
    });
}

// 틴친 프로필 모달 닫기 이벤트
const teenchinBox = document.querySelector(".teenchin-box");

if (teenchinBox && commentProfileImg){
    document.addEventListener("click", (e) => {
        if (!commentProfileImg.contains(e.target) && !teenchinBox.contains(e.target)) {
            profile.classList.add("hidden");
        }
    });
}



sendLetterBoxBtn.addEventListener("click", () => {
    profile.classList.add("hidden");
    sendLetter.classList.remove("hidden");
})

// 쪽지 보내기 닫기(버튼) 모달 이벤트
const sendLetterCloseBtn = document.querySelector(".send-close-btn");

if (sendLetterCloseBtn){
    sendLetterCloseBtn.addEventListener("click", () => {
        sendLetter.classList.add("hidden");
    });
}

// 쪽지 보내기 닫기(여백) 모달 이벤트
const sendLetterModal = document.querySelector(".send-modal-box");

if (sendLetterModal){
    document.addEventListener("click", (e) => {
        if (!sendLetterBoxBtn.contains(e.target) && !sendLetterModal.contains(e.target)) {
            sendLetter.classList.add("hidden");
        }
    });
}

// 쪽지 보내기 모달 이벤트
const sendLetterBtn = document.querySelector(".send-check-btn");

if (sendLetterBtn){
    sendLetterBtn.addEventListener("click", async () => {
        const letterContent = document.getElementById("letter-content").value;
        if (!letterContent) return;
        const receiverId = document.querySelector(".send-receiver-email").innerText;
        await activityLetterService.write({
            letter_content: letterContent,
            receiver_id: receiverId
        })
        Swal.fire("쪽지가 전송 되었습니다.", "", "success");
    });
}

// 틴친 추가 모달 이벤트
const teenFriendAdd = document.querySelector(".teenchin-add-btn");
const teenFriendRequest = document.querySelector(".teenchin-request-btn");

if (teenFriendAdd){
    teenFriendAdd.addEventListener("click", () => {
        Swal.fire({
            title: "틴친 신청을 보낼까요?",
            text: "",
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#CE201B",
            cancelButtonColor: "#E1E1E1",
            confirmButtonText: "친구추가",
            cancelButtonText: "닫기",
        }).then(async (result) => {
            if (result.value) {
                await activityTeenchinService.apply(opponentTeenchinId);
                await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
                Swal.fire("틴친 신청을 보냈어요!", "", "success");
            } else if ((result.dismiss = "cancel")) {
                return;
            }
        });
    });
}

// 틴친 신청 취소 모달 이벤트
if (teenFriendRequest){
    teenFriendRequest.addEventListener("click", () => {
        Swal.fire({
            title: "신청을 취소할까요?",
            text: "",
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#CE201B",
            cancelButtonColor: "#E1E1E1",
            confirmButtonText: "신청취소",
            cancelButtonText: "닫기",
        }).then(async (result) => {
            if (result.value) {
                await activityTeenchinService.cancelOrAcceptDenyTeenchin(opponentTeenchinId, true, false);
                await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
                Swal.fire("틴친 신청이 취소되었습니다.", "", "success");
            } else if ((result.dismiss = "cancel")) {
                return;
            }
        });
    });
}

// 틴친 수락/거절 모달 이벤트
if (teenchinAcceptButton) {
    teenchinAcceptButton.addEventListener("click", () => {
        Swal.fire({
            title: "신청을 수락할까요?",
            text: "",
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#CE201B",
            cancelButtonColor: "#E1E1E1",
            confirmButtonText: "수락",
            cancelButtonText: "거절",
        }).then(async (result) => {
            if (result.value) {
                await activityTeenchinService.cancelOrAcceptDenyTeenchin(opponentTeenchinId, false, true);
                await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
                Swal.fire("틴친신청을 수락했습니다.", "", "success");
            } else if ((result.dismiss = "cancel")) {
                await activityTeenchinService.cancelOrAcceptDenyTeenchin(opponentTeenchinId, false, false);
                await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
                Swal.fire("틴친신청을 거절했습니다.", "", "success");
                return;
            }
        });
    })
}

// 틴친 취소 모달 이벤트
const teenFriendCancle = document.querySelector(".teenchin-btn");

if (teenFriendCancle){
    teenFriendCancle.addEventListener("click", () => {
        Swal.fire({
            title: "틴친을 그만둘까요?",
            text: "",
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#CE201B",
            cancelButtonColor: "#E1E1E1",
            confirmButtonText: "틴친끊기",
            cancelButtonText: "닫기",
        }).then(async (result) => {
            if (result.value) {
                await activityTeenchinService.removeTeenchin(opponentTeenchinId);
                await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
                Swal.fire("틴친 관계가 해제되었어요.", "", "success");
            } else if ((result.dismiss = "cancel")) {
                return;
            }
        });
    });
}

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
let page = 1;
// 댓글 더보기 버튼 표시 여부
const replyMoreButton = document.querySelector(".show-more-btn-wrap");
const showOrHideMoreButton = (isAdd, replies) => {
    if (replies.length === 0){
        replyMoreButton.style.display = "none";
    } else {
        replyMoreButton.style.display = "block";
    }
}

// 댓글 옆 버튼 클릭 시 수정/삭제 메뉴 열기
const bgForModalClose = document.querySelector(".bg-for-modal")
const showMenuButtons = () => {
    const replyMenuShowButtons = document.querySelectorAll(".comment-modify-button")
    const replyMenuButtons = document.querySelectorAll(".k-comment-menu-open-wrap")
    replyMenuShowButtons.forEach((btn) => {
        btn.addEventListener("click", (e) => {
            let replyId = btn.classList[1];
            replyMenuButtons.forEach((replyBtn) => {
                if (replyBtn.classList[1] === replyId) {
                    replyBtn.style.display = "block";
                } else {
                    replyBtn.style.display = "none";
                }
                bgForModalClose.style.display = "block";
                bgForModalClose.addEventListener("click", (e) => {
                    if (e.target !== replyBtn && !replyBtn.contains(e.target)) {
                        replyBtn.style.display = "none";
                    }
                    bgForModalClose.style.display = "none";
                })
            })
        })
    })
}

// 수정 메뉴 클릭 시 수정창 출력 이벤트 등록
const addClickEventUpdate = () => {
    let updateOpenBtns = document.querySelectorAll(".k-comment-menu-open-choice.update")
    let updateModals = document.querySelectorAll(".k-comment-update-box-all-wrap")
    let originalReplies = document.querySelectorAll(".k-comment-list-wrap")

    updateOpenBtns.forEach((btn) => {
        btn.addEventListener("click", (e) => {
            updateModals.filter((modal) => modal.classList[1] === e.target.classList[2])
                .map((modal) => {
                    modal.style.display = "block";
                    let originalReplyContents = document.querySelector(`.text${e.target.classList[2]}`)
                    let updateTextareas = document.getElementById(`textarea${e.target.classList[2]}`)
                    updateTextareas.value = originalReplyContents.innerText;
                });
            originalReplies.filter((reply) => reply.classList[1] === e.target.classList[2])
                .map((reply) => {
                    reply.style.display = "none";
                })
            bgForModalClose.style.display = "none";
        })
    })
}

// 수정창에서 취소 버튼 클릭 시 수정창 닫기 이벤트 등록
const addClickEventHideUpdate = () => {
    const updateCancelButtons = document.querySelectorAll(".k-comment-update-close-button")
    const updateModals = document.querySelectorAll(".k-comment-update-box-all-wrap");
    const originalReplies = document.querySelectorAll(".k-comment-list-wrap")
    updateCancelButtons.forEach((btn) => {
        btn.addEventListener("click", (e) => {
            updateModals.filter((modal) => modal.classList[1] === e.target.classList[1])
                .map((modal, i) => {
                    modal.style.display = "none";
                })
            originalReplies.filter((reply) => reply.classList[1] === e.target.classList[1])
                .map((reply) => {
                    reply.style.display = "flex";
                })
            bgForModalClose.style.display = "none";
        })
    })
}

activityReplyService.getList(true, page+1, activityId, showOrHideMoreButton);

// 댓글 목록 불러오기, 댓글 작성, 댓글 수정/삭제
const commentWrap = document.querySelector(".k-comment-list-box-wrap")
const memberId = document.querySelector("input[name=member-id]").value;
const showReplies = async (isAdd, replies) => {
    if (replies.length === 0) {
        commentWrap.innerHTML = `
            <div class="k-comment-line"></div>
            <div class="k-comment-list-all-wrap"></div>
            <div class="feed-item">
                <span>등록된 댓글이 없습니다.</span>
            </div>
        `;
    } else {
        const commentListAllWrap = document.querySelector(".k-comment-list-all-wrap")
        let emptyCommentWrap = document.querySelector(".k-comment-list-box-wrap .feed-item");
        if (emptyCommentWrap) {
            emptyCommentWrap.style.display = "none";
        }
        let text = ``;
        replies.forEach((reply) => {
            text += `
                <!-- 댓글 수정 부분 -->
                <div class="k-comment-update-box-all-wrap ${reply.id}">
                    <div class="k-comment-update-box-wrap">
                        <div class="k-comment-update-wrap">
                            <div class="k-comment-update-username-container">${reply.member_nickname}</div>
                            <div class="k-comment-update-container">
                                <textarea class="k-comment-update-guide" id="textarea${reply.id}" name="reply-content" type="text" placeholder="수정할 댓글을 남겨주세요. 욕설, 비방글은 무통보 삭제됩니다." autocomplete="off" required=""></textarea>
                                <div class="k-comment-update-upload-wrap">
                                    <div class="k-comment-update-upload-container">
                                        <button class="k-comment-update-upload-button ${reply.id}" type="button">수정</button>
                                        <button class="k-comment-update-close-button ${reply.id}" type="button">취소</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <!-- 개별 댓글 부분 -->
                <div class="k-comment-list-wrap ${reply.id}">
                <input type="hidden" class="member-email${reply.id}" name="writer-email" value="${reply.member_email}">
                    <!-- 댓글 내 프로필 사진 부분 -->
                    <div class="k-comment-profile-container ${reply.id}">
                        <img src="${reply.member_path ? '/upload/' + reply.member_path : '/static/public/web/images/logo/logo1.png'}" alt="프로필사진" class="k-comment-profile-icon profile-image${reply.id}">
                    </div>
                    <!-- 개별 댓글 전체 내용 부분 -->
                    <div class="k-comment-content-container">
                        <!-- 댓글 정보 부분 -->
                        <div class="k-comment-info">
                            <!-- 댓글 작성자 id -->
                            <input type="hidden" name="writer-id" class="reply-writer-id${reply.id}" value="${reply.member_id}">
                            <!-- 댓글 작성자 이름 부분 -->
                            <span class="member-name${reply.id}">${reply.member_nickname}</span>
                            <!-- 댓글 작성 날짜 부분 -->
                            <span class="k-comment-info-date">${timeForToday(reply.created_date)}</span>
                        </div>
                        <!-- 개별 댓글 내용 부분 -->
                        <div class="k-comment-text text${reply.id}">${reply.reply_content.replace(/\n/g, '<br>')}</div>
                    </div>
                    <div class="comment-modify-button ${reply.id}" style="display: ${reply.member_id !== Number(memberId) ? 'none' : 'block'};">
                        <button class="k-comment-menu" type="button">
                            <svg class="k-comment-menu-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                                <path fill-rule="evenodd" d="M10.5 6a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0zm0 6a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0zm0 6a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0z" clip-rule="evenodd"></path>
                            </svg>
                        </button>
                        <div class="k-comment-menu-open-wrap ${reply.id}">
                            <div class="k-comment-menu-open-container" role="none">
                                <div class="k-comment-menu-open-divison" role="none">
                                    <button class="k-comment-menu-open-choice update ${reply.id}" type="button" id="comment-menu-open-update">수정</button>
                                    <button class="k-comment-menu-open-choice delete ${reply.id}" type="button" id="comment-menu-open-delete">삭제</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                `
        })
        if (isAdd){
            commentListAllWrap.innerHTML += text;
        } else {
            commentListAllWrap.innerHTML = text;
        }
        await activityReplyService.getList(true, page+1, activityId, showOrHideMoreButton);
        showMenuButtons();
        addClickEventUpdate();
        addClickEventHideUpdate();
        addClickEventUpdateUpload();
        addClickEventDelete();
        addClickEventReplyProfile();
    }
}

replyMoreButton.addEventListener("click", async () => {
    page++;
    await activityReplyService.getList(true, page, activityId, showReplies);
    await activityReplyService.getList(true, page+1, activityId, showOrHideMoreButton);
})

function timeForToday(datetime) {
    const today = new Date();
    const date = new Date(datetime);

    let gap = Math.floor((today.getTime() - date.getTime()) / 1000 / 60);

    if (gap < 1) {
        return "방금 전";
    }

    if (gap < 60) {
        return `${gap}분 전`;
    }

    gap = Math.floor(gap / 60);

    if (gap < 24) {
        return `${gap}시간 전`;
    }

    gap = Math.floor(gap / 24);

    if (gap < 31) {
        return `${gap}일 전`;
    }

    gap = Math.floor(gap / 31);

    if (gap < 12) {
        return `${gap}개월 전`;
    }

    gap = Math.floor(gap / 12);

    return `${gap}년 전`;
}



// 댓글 작성하기
const replyUploadButton = document.getElementById("comment-upload-button")
replyUploadButton.addEventListener("click", async () => {
    const replyTextarea = document.querySelector("textarea.k-comment-input-guide")
    const replyText = replyTextarea.value;
    if (!replyText) return;
    await activityReplyService.write({
        'member_id': memberId,
        'reply_content': replyText,
        'activity_id': activityId
    });
    page = 1;
    replyTextarea.value = '';
    await activityReplyService.getList(false, page, activityId, showReplies);
})

// 댓글 수정하기
const addClickEventUpdateUpload = () => {
    const replyUpdateUploadButtons = document.querySelectorAll(".k-comment-update-upload-button");
    replyUpdateUploadButtons.forEach((btn) => {
        btn.addEventListener("click", async (e) => {
            const updateTextarea = document.getElementById(`textarea${e.target.classList[1]}`);
            const updateContent = updateTextarea.value;
            if (!updateContent) return;
            await activityReplyService.update({
                'member_id': memberId,
                'reply_content': updateContent,
                'activity_id': activityId,
                'id': e.target.classList[1]
            });
            page = 1;
            updateTextarea.value = '';
            await activityReplyService.getList(false, page, activityId, showReplies);
        })})}

// 댓글 삭제하기
const addClickEventDelete = () => {
    const replyDeleteButtons = document.querySelectorAll(".k-comment-menu-open-choice.delete")
    replyDeleteButtons.forEach((btn) => {
        btn.addEventListener("click", async (e) => {
            const id = e.target.classList[2];
            await activityReplyService.remove({
                'id': id
            });
            page = 1;
            await activityReplyService.getList(false, page, activityId, showReplies);
        })
    })
}

activityReplyService.getList(true, page, activityId, showReplies);


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// 공유하기 버튼
const shareBtn = document.getElementById("share-button");
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

// 모임 공지사항 불러오기(모임공지 쪽 작업 끝나면 추가 예정)
// const getClubNoticeList = async (callback) => {
//     const clubId = document.querySelector("input[name='club-id']").value
//     const response = await fetch('/')
// }

// 공지사항 각각 제목 클릭 시 세부 내용 표시
const noticeContentWraps = document.querySelectorAll(".club-notice-content-wrap");
const noticeTitles = document.querySelectorAll(".club-notice-box");
const noticeShowBtns = document.querySelectorAll(".club-notice-show-icon");
const noticeHideBtns = document.querySelectorAll(".club-notice-hide-icon");

noticeTitles.forEach((title, i) => {
    title.addEventListener("click", () => {
        if (noticeShowBtns[i].style.display == "block") {
            noticeShowBtns[i].style.display = "none";
            noticeHideBtns[i].style.display = "block";
        } else {
            noticeShowBtns[i].style.display = "block";
            noticeHideBtns[i].style.display = "none";
        }
        if (noticeContentWraps[i].style.display == "none") {
            noticeContentWraps[i].style.display = "block";
        } else {
            noticeContentWraps[i].style.display = "none";
        }
    });
});

// 활동 카테고리 클릭 시 활동 목록에 해당 카테고리 검색 결과로 이동
const activityDetailCategoryForm = document.querySelector(".activity-detail-category-form");
activityDetailCategoryForm.addEventListener("click", () => {
    activityDetailCategoryForm.submit();
})

// 신청하기 버튼 클릭 시 신청페이지로 이동
const activityJoinButton = document.querySelector(".real-button");
activityJoinButton.addEventListener("click", () => {
    location.href = `/activity/join?id=${activityId}`;
})

// 관리하기 버튼 클릭 시 관리페이지로 이동
const activityManageButton = document.querySelector(".manage-item-button");
activityManageButton.addEventListener("click", () => {
    location.href = `/member/activity?activity_id=${activityId}`;
})