// 태그 검색 이벤트
// 해시태그 검색 묶음 쿼리
const inputTagsSearchBorder = document.querySelector(".tag-search");
// 해시태그 검색 input 쿼리
const inputTagsSearch = document.querySelector(".tag-search-text");
// 해시태그 검색 아이콘 button 쿼리
const inputTagsSearchIcon = document.querySelector(".tag-search-icon");

// 해시태그 검색 값 입력 시 이벤트 발생
// 값이 없으면, 박스 및 버튼 등 색상 붉은 색으로 변경
// 값이 있으면 다시 정상 출력
inputTagsSearch.addEventListener("keyup", (e) => {
    if (!e.target.value) {
        inputTagsSearch.classList.add("red")
        inputTagsSearchIcon.style = "color: #CE201B";
        inputTagsSearchBorder.style = "border-color: #CE201B";
        return;
    } else {
        inputTagsSearch.classList.remove("red")
        inputTagsSearchIcon.style = "color: #878D91";
        inputTagsSearchBorder.style = "border-color: #E1E4E6";
    }
});



// 상단 카테고리 선택 이벤트
// 각 카테고리 button 쿼리
const wishlistItems = document.querySelectorAll(".top-categroy-item");

// 각 카테고리 버튼 클릭 시 이벤트 발생
// 특정 카테고리 선택 시 전체 카테고리의 선택 속성 해제 후
// 선택한 카테고리에만 선택 속성 부여
wishlistItems.forEach((button) => {
    button.addEventListener('click', () => {
        wishlistItems.forEach((item) => {
            item.classList.remove("all");
        });

        button.classList.add("all");
    });
});



// 위시리스트 게시글 좋아요 클릭 이벤트
// 하트 아이콘 쿼리
const likeIcon = document.querySelector(".post-like-icon");
// 하트 개수 쿼리
const likeCount = document.querySelector(".post-like-count");

// 하트 아이콘 클릭 시 이벤트 발생
// 좋아요가 눌렀다면, 좋아요 취소 후 개수 -1
// 좋아요를 누르지 않았다면, 좋아요 표기 후 개수 +1
likeIcon.addEventListener("click", () => {
    if (likeIcon.classList.contains('liked')) {
        likeIcon.classList.remove("liked");
        likeCount.textContent = parseInt(likeCount.textContent) - 1;
    } else {
        likeIcon.classList.add("liked");
        likeCount.textContent = parseInt(likeCount.textContent) + 1;
    }
});



// 위시리스트 게시글 댓글 카운팅 이벤트
// 개별 댓글 묶음 쿼리
const commentList = document.querySelectorAll(".comment-list-all-wrap");
// 게시글 댓글 개수 쿼리
const commentCount =  document.querySelector(".post-comment-count");



// 개별 댓글 묶음 개수(길이) = 게시글 댓글 개수로 표기
function commentCountNum() {
    commentCount.textContent = commentList.length;
}

commentCountNum();



// 위시리스트 게시글 메뉴 열고 닫기 이벤트
// 위시리스트 게시글 내 메뉴 버튼 쿼리
const wishlistPostMenuButton = document.querySelector(".post-menu-container");
// 위시리스트 게시글 메뉴(기본 가려짐) 쿼리
const wishlistPostMenu = document.querySelector(".post-modal-wrap");

// 위시리스트 게시글 내 메뉴 버튼 클릭 시 클릭 이벤트 발생
// 위시리스트 게시글 메뉴를 활성화 or 비활성화
wishlistPostMenuButton.addEventListener("click", () => {
    wishlistPostMenu.classList.toggle("hidden");
});



// 위시리스트 메뉴 닫기 이벤트
const wishlistPostMenuBox = document.querySelector(".post-menu-open-container");

// 여백 클릭 시 위시리스트 메뉴 자동 숨기기
document.addEventListener("click", (e) => {
    if (!wishlistPostMenuButton.contains(e.target) && !wishlistPostMenuBox.contains(e.target)) {
        wishlistPostMenu.classList.add("hidden");
    }
});



// 위시리스트 게시글 수정 모달창
// 위시리스트 게시글 메뉴 중 수정 버튼 쿼리
const postUpdateButton = document.getElementById("post-menu-open-update");
// 위시리스트 게시글 수정 모달 쿼리
const modalPostUpdate = document.querySelector(".post-update");
// 위시리스트 게시글 수정 모달 돌아가기 버튼 쿼리
const modalPostUpdateClose = document.querySelector(".update-close-container");
// 위시리스트 게시글 수정 모달 발행하기 버튼 쿼리
const modalPostUpdateFinish = document.querySelector(".update-finish-botton");

// 위시리스트 수정 메뉴 버튼 클릭 시 이벤트 발생
// 위시리스트 수정 모달 보이기
postUpdateButton.addEventListener("click", () => {
    wishlistPostMenu.classList.add("hidden");
    modalPostUpdate.classList.remove("hidden");
});

// 위시리스트 수정 모달 돌아가기 버튼 클릭 시 이벤트 발생
// 위시리스트 수정 모달 숨기기
modalPostUpdateClose.addEventListener("click", () => {
    modalPostUpdate.classList.add("hidden");
});

// 위시리스트 수정 모달 발행하기 버튼 클릭 시 이벤트 발생
// 위시리스트 수정 모달 숨기기
modalPostUpdateFinish.addEventListener("click", () => {
    modalPostUpdate.classList.add("hidden");
});



// 위시리스트 게시글 수정 타이틀 입력 이벤트
// 위시리스트 게시글 수정 모달 제목 입력 쿼리
const updatePostTitle = document.querySelector(".update-category-input");

// 위시리스트 게시글 수정 모달 발행하기 버튼 클릭 시 이벤트 발생
// 수정 모달에 타이틀이 입력되지 않으면 안내 문구 빨간색으로 출력
modalPostUpdateFinish.addEventListener("click", () => {
    if (!updatePostTitle.value) {
        updatePostTitle.classList.add("red");
        return;
    } else {
        updatePostTitle.classList.remove("red");
    }
});

// 위시리스트 게시글 태그 수정 이벤트
// 위시리스트 수정 모달 태그 입력 쿼리
const tagInput = document.querySelector(".update-tags-input-container .update-tags-input");
// 위시리스트 수정 모달 태그 묶음 쿼리
const tagWrap = document.querySelector(".update-tags-wrap");
// 위시리스트 수정 모달 태그 목록 쿼리
const tag = document.querySelector(".update-tags-wrap .tag");
// 위시리스트 수정 모달 태그 에러 메시지 쿼리
const tagError = document.querySelector(".tag-error-text");

// 위시리스트 수정 모달에서 태그 입력 시 이벤트 발생
// 만약, 엔터를 입력했다면,
tagInput.addEventListener("keyup", (e) => {
    if (e.keyCode === 13) {
        // 위시리스트 수정 모달 태그 목록 쿼리
        const tags = document.querySelectorAll(".update-tags-wrap .tag-list");

        // 입력 값의 길이가 10 이하이며, 목록의 태그 개수가 5이하면,
        // 태그 값 추가(태그 생성)
        if (e.target.value.length <= 10 && tags.length <= 5) {
            if (e.target.value) {
                const tagDiv = document.createElement("div");
                tagDiv.classList.add("tag-list");
                tagDiv.innerHTML = `<span>${e.target.value} ×</span>`;
                tag.appendChild(tagDiv);
                e.target.value = "";
                tagError.classList.add("hidden")

                // tagDiv 클릭 시 이벤트 발생
                tagDiv.addEventListener("click", (e) => {
                    // 태그 삭제
                    e.stopPropagation();
                    tag.removeChild(tagDiv);
                    tagError.classList.add("hidden")
                });
            } 
        } else {
            // 위 값이 아니라면, 오류 메시지 출력
            tagError.classList.remove("hidden")
        }
    }
});



// 위시리스트 게시글 수정 모달창 닫기 이벤트
// 위시리스트 게시글 수정 모달 묶음 쿼리
const postUpdateWrap = document.querySelector(".post-update-wrap");

// 화면 클릭 시 위시리스트 게시글 수정 모달 숨기기
document.addEventListener("click", (e) => {
    if (!postUpdateButton.contains(e.target) && !postUpdateWrap.contains(e.target)) {
        modalPostUpdate.classList.add("hidden");
    }
});



// 위시리스트 게시글 댓글 열고 닫기 이벤트
// 위시리스트 댓글 열기 쿼리
const commentOpen = document.querySelector(".comment-open-wrap");
// 위시리스트 게시글 닫기 쿼리
const commentClose = document.querySelector(".comment-close-wrap");
// 위시리스트 댓글 쿼리
const comment = document.querySelector(".comment-all-wrap");

// 위시리스트 댓글 열기 클릭 시 이벤트 발생
// 댓글 열기 버튼 비활성화 > 닫기 버튼, 댓글 목록 활성화
commentOpen.addEventListener("click", () => {
    commentOpen.classList.add("hidden");
    commentClose.classList.remove("hidden");
    comment.classList.remove("hidden");
});

// 위시리스트 게시글 댓글 닫기 버튼 클릭 시 이벤트 발생
// 댓글 닫기 버튼, 댓글 목록 비활성화 > 댓글 열기 버튼 활성화
commentClose.addEventListener("click", () => {
    commentOpen.classList.remove("hidden");
    commentClose.classList.add("hidden");
    comment.classList.add("hidden");
});



// 위시리스트 댓글 메뉴 열고 닫기 이벤트
// 위시리스트 댓글 내 메뉴 버튼 쿼리
const wishlistCommentMenuButton = document.querySelector(".comment-menu");
// 위시리스트 댓글 메뉴 목록 쿼리
const wishlistCommentMenu = document.querySelector(".comment-modal-wrap");

// 위시리스트 댓글 버튼 클릭 시 이벤트 발생
// 위시리스트 댓글 메뉴를 활성화 or 비활성화
wishlistCommentMenuButton.addEventListener("click", () => {
    wishlistCommentMenu.classList.toggle("hidden");
});




// 위시리스트 댓글 메뉴 닫기 이벤트
// 
const commentMenuOpenContainer = document.querySelector(".comment-menu-open-container");

// 위시리스트 댓글 메뉴 닫기 이벤트
document.addEventListener("click", (e) => {
    if (!wishlistCommentMenuButton.contains(e.target) && !commentMenuOpenContainer.contains(e.target)) {
        wishlistCommentMenu.classList.add("hidden");
    }
});



// 댓글 수정 이벤트
// 댓글 메뉴 중 수정하기 버튼 쿼리
const commentMenuOpenUpdate = document.getElementById("comment-menu-open-update");
// 댓글 수정 입력 쿼리
const commentInputUpdate = document.querySelector(".comment-update-box-all-wrap");
// 개별 댓글 쿼리
const commentComment = document.querySelector(".comment-list-all-wrap");

// 댓글 메뉴 중 수정하기 버튼 클릭 시 이벤트 발생
// 댓글 수정 입력 활성화 후 개별 댓글 숨기기
commentMenuOpenUpdate.addEventListener("click", () => {
    commentInputUpdate.classList.remove("hidden");
    commentComment.classList.add("hidden");
    wishlistCommentMenu.classList.add("hidden");
});

// 댓글 수정 등록 버튼
const commentUploadFinish =document.getElementById("comment-update-upload");

// 댓글 수정 등록 버튼 클릭 시 이벤트 발생
// 댓글 수정 입력 비활성화 개별 댓글 활성화
commentUploadFinish.addEventListener("click", () => {
    commentInputUpdate.classList.add("hidden");
    commentComment.classList.remove("hidden");
});



// 위시리스트 생성 모달창
// const wishlistCreate = document.querySelector(".main-extra-create-wrap")

// 위시리스트 생성 모달창 - 미로그인 시 발생 이벤트
// const modalLogin = document.querySelector(".modal-login")
// const modalLoginClose = document.querySelector(".modal-divison-login-button-close")
// const modalLoginFinish = document.querySelector(".modal-divison-login-button")

// wishlistCreate.addEventListener("click", () => {
//     modalLogin.classList.remove("hidden");
// });

// modalLoginClose.addEventListener("click", () => {
//     modalLogin.classList.add("hidden");
// });

// modalLoginFinish.addEventListener("click", () => {
//     modalLogin.classList.add("hidden");
// });



// 위시리스트 생성 모달창 - 정상 로그인
// 위시리스트 생성 버튼 쿼리
const wishlistCreate = document.querySelector(".extra-create-button")

// 위시리스트 생성 모달 쿼리
const modalCreateInput = document.querySelector(".post-create")
// 위시리스트 생성 모달 돌아가기 버튼 쿼리
const modalCreateClose = document.querySelector(".create-close-container")
// 위시리스트 생성 모달 발행하기 버튼 쿼리
const modalCreateFinish = document.querySelector(".create-finish-container")

// 위시리스트 생성 모달창 열기 이벤트
// 위시리스트 생성 버튼 클릭 시 생성 모달 활성화 이벤트 발생
wishlistCreate.addEventListener("click", () => {
    modalCreateInput.classList.remove("hidden");
});

// 위시리스트 생성 모달창 닫기(아이콘) 이벤트
// 위시리스트 돌아가기 버튼 클릭 시 생성 모달 비활성화 이벤트 발생
modalCreateClose.addEventListener("click", () => {
    modalCreateInput.classList.add("hidden");
});

// 위시리스트 생성 모달창 완료 이벤트
// 위시리스트 발행하기 버튼 클릭 시 생성 모달 비활성화 이벤트 발생
modalCreateFinish.addEventListener("click", () => {
    modalCreateInput.classList.add("hidden");
});



// 위시리스트 게시글 태그 추가 이벤트
// 위시리스트 생성 모달 태그 입력 쿼리
const createTagInput = document.querySelector(".create-tags-input-container .create-tags-input");
// 위시리스트 생성 모달 태그 묶음 쿼리
const createTagWrap = document.querySelector(".create-tags-wrap");
// 위시리스트 생성 모달 태그 목록 쿼리
const createTag = document.querySelector(".create-tags-wrap .create-tag");
// 위시리스트 생성 모달 에러 메시지 쿼리
const createTagError = document.querySelector(".create-tag-error");

// 위시리스트 생성 모달 태그 입력 시 이벤트 발생
createTagInput.addEventListener("keyup", (e) => {
    // 만약, enter 입력 시
    if (e.keyCode === 13) {
        // 위리시리스트 생성 모달 태그 목록 쿼리
        const createTags = document.querySelectorAll(".create-tags-wrap .create-tag-list");

        // 입력 값의 길이가 10 이하이며, 목록의 태그 개수가 5이하면,
        // 태그 값 추가(태그 생성)
        if (e.target.value.length <= 10 && createTags.length <= 5) {
            if (e.target.value) {
                const createTagDiv = document.createElement("div");
                createTagDiv.classList.add("create-tag-list");
                createTagDiv.innerHTML = `<span>${e.target.value} ×</span>`;
                createTag.appendChild(createTagDiv);
                e.target.value = "";
                createTagError.classList.add("hidden")

                // createTagDiv 클릭 시 태그 삭제 이벤트 발생
                createTagDiv.addEventListener("click", (e) => {
                    // 취소
                    e.stopPropagation();
                    createTag.removeChild(createTagDiv);
                    createTagError.classList.add("hidden")
                });
            } 
        // 위가 아니라면 에러 메시지 출력
        } else {
            createTagError.classList.remove("hidden")
        }
    }
});



// 위시리스트 생성 모달창 닫기(여백) 이벤트
// 위시리스트 생성 모달 묶음 쿼리
const postCreateWrap = document.querySelector(".post-create-wrap");

// 여백 클릭 시 생성 모달 비활성화 이벤트 발생
document.addEventListener("click", (e) => {
    if (!wishlistCreate.contains(e.target) && !postCreateWrap.contains(e.target)) {
        modalCreateInput.classList.add("hidden");
    }
});



// 프로필 클릭 시 틴친 프로필 모달 출력 이벤트
// 게시글 내 틴친 프로필 사진 쿼리
const postProfileImg = document.querySelector(".post-profile-img-container")
// 댓글 내 틴친 프로필 사진 쿼리
const commentProfileImg = document.querySelector(".comment-profile-container")
// 틴친 프로필 모달 묶음 쿼리
const profile = document.querySelector(".profile")

// 게시글 내 틴친 프로필 사진 쿼리 클릭 시 프로필 모달 활성화 이벤트 발생
postProfileImg.addEventListener("click", () => {
    profile.classList.remove("hidden");
});

// 댓글 내 틴친 프로필 사진 쿼리 클릭 시 프로필 모달 활성화 이벤트 발생
commentProfileImg.addEventListener("click", () => {
    profile.classList.remove("hidden");
});




// 틴친 프로필 모달 닫기 이벤트
// 틴친 프로필 모달 쿼리
const teenchinBox = document.querySelector(".teenchin-box")

// 여백 클릭시 틴친 프로필 모달 쿼리 비활성화 이벤트 발생
document.addEventListener("click", (e) => {
    if (!postProfileImg.contains(e.target) && !commentProfileImg.contains(e.target) && !teenchinBox.contains(e.target)) {
        profile.classList.add("hidden");
    }
});



// 쪽지 보내기 클릭 시 쪽지 보내기 모달 출력 이벤트
// 쪽지 보내기 버튼 쿼리
const sendLetterBoxBtn = document.querySelector(".send-letter-btn")
// 쪽지 보내기 모달 쿼리
const sendLetter = document.querySelector(".send-modal-wrap")

// 쪽지 보내기 버튼 클릭 시 이벤트 발생
// 틴친 프로필 모달 비활성화 및 쪽지 보내기 모달 활성화
sendLetterBoxBtn.addEventListener("click", () => {
    profile.classList.add("hidden");
    sendLetter.classList.remove("hidden");
});



// 쪽지 보내기 닫기(버튼) 모달 이벤트
// 쪽지 보내기 닫기 버튼 쿼리
const sendLetterCloseBtn = document.querySelector(".send-close-btn")

// 쪽지 보내기 닫기 버튼 클릭 시 쪽지 보내기 모달 비활성화
sendLetterCloseBtn.addEventListener("click", () => {
    sendLetter.classList.add("hidden");
});




// 쪽지 보내기 닫기(여백) 모달 이벤트
// 여백 클릭 시 쪽지 보내기 창 닫기 이벤트 발생
const sendLetterModal = document.querySelector(".send-modal-box")

document.addEventListener("click", (e) => {
    if (!sendLetterBoxBtn.contains(e.target) && !sendLetterModal.contains(e.target)) {
        sendLetter.classList.add("hidden");
    }
});



// 쪽지 보내기 모달 이벤트
// 쪽지 보내기 버튼 쿼리
const sendLetterBtn = document.querySelector(".send-check-btn")

// 쪽지 보내기 버튼 클릭 시 쪽지 발송 완료 모달 활성화
sendLetterBtn.addEventListener("click", () => {
    Swal.fire({
        title: "쪽지가 전송 되었습니다.",
        text: "",
        icon: "success",
        confirmButtonColor: "#CE201B"
    });
});


// 틴친 추가 모달 이벤트
// 틴친 추가 버튼 쿼리
const teenFriendAdd = document.querySelector(".teenchin-add-btn")
// 틴친 대기 중 버튼 쿼리
const teenFriendRequest = document.querySelector(".teenchin-request-btn")

// 틴친 추가 버튼 클릭 시 틴친 신청 확인 모달 활성화
// 닫기 시 모달 종료, 추가 시 틴친 대기중 버튼 활성화
teenFriendAdd.addEventListener("click", () => {
    Swal.fire({
        title: "틴친 신청을 보낼까요?",
        text: "",
        icon: "question",
        showCancelButton: true,
        confirmButtonColor: "#CE201B",
        cancelButtonColor: "#E1E1E1",
        confirmButtonText: "친구추가",
        cancelButtonText: "닫기",
    }).then((result) => {
        if (result.value) {
            // 틴플레이 삭제 관련 서버 작업 코드 입력
            // 완료 시 아래 코드 실행 (실제로는 또 .then(()=>{}) 으로 써야함)
            teenFriendAdd.classList.add("hidden");
            teenFriendRequest.classList.remove("hidden");
        } else if ((result.dismiss = "cancel")) {
            return;
        }
    });
});

// 틴친 대기중 버튼 클릭 시 신청 취소 확인 모달 활성화
// 닫기 시 모달 종료, 신청 취소 시 틴친 추가 버튼 활성화
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
    }).then((result) => {
        if (result.value) {
            // 틴플레이 삭제 관련 서버 작업 코드 입력
            // 완료 시 아래 코드 실행 (실제로는 또 .then(()=>{}) 으로 써야함)
            teenFriendRequest.classList.add("hidden");
            teenFriendAdd.classList.remove("hidden");
        } else if ((result.dismiss = "cancel")) {
            return;
        }
    });
})

// 틴친 취소 모달 이벤트
// 틴친중 버튼 쿼리
const teenFriendCancle = document.querySelector(".teenchin-btn");

// 틴친중 버튼 클릭 시 틴친 그만두기 모달 활성화 이벤트 발생
// 닫기 클릭 시 모달 종료, 틴친 끊기 클릭 시 틴친 추가 버튼 활성화
teenFriendCancle.addEventListener("click", () => {
    Swal.fire({
        title: "틴친을 그만둘까요?",
        text: "",
        icon: "error",
        showCancelButton: true,
        confirmButtonColor: "#CE201B",
        cancelButtonColor: "#E1E1E1",
        confirmButtonText: "틴친끊기",
        cancelButtonText: "닫기",
    }).then((result) => {
        if (result.value) {
            // 틴플레이 삭제 관련 서버 작업 코드 입력
            // 완료 시 아래 코드 실행 (실제로는 또 .then(()=>{}) 으로 써야함)
            teenFriendCancle.classList.add("hidden");
            teenFriendAdd.classList.remove("hidden");
        } else if ((result.dismiss = "cancel")) {
            return;
        }
    });
})