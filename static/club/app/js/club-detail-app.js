// 처음에 페이지 로드 시 9번째 활동부터는(존재한다면) 숨겨놓기
const finishedActivities = document.querySelectorAll(".finished-events-boxes");
if (finishedActivities.length >= 9) {
    for (let i = 8; i < finishedActivities.length; i++) {
        finishedActivities[i].style.display = "none";
    }
}

// 처음에 게시물이 8개 이하라면 더보기 버튼 숨기기
const moreBtn = document.querySelector(".show-more-btn");
if (finishedActivities.length < 9) {
    moreBtn.style.display = "none";
}

// 더보기 버튼 클릭 시 8개 더 표시 후 마지막이라면 더보기 버튼도 없애기
// 이때 더보기 버튼 안의 숫자도 바꿔주어야 한다.
// 먼저 현재 몇 개 표시되어 있는지를 구해주는 함수를 분리하여 작성해준다.(재사용을 위해)
function getShownCounts() {
    let currentShownCount = 0;
    for (let i = 0; i < finishedActivities.length; i++) {
        if (finishedActivities[i].style.display === "none") {
            currentShownCount = i;
            break;
        }
    }
    return currentShownCount;
}

const moreCount = document.querySelector(".show-more-counts");
// 우선 페이지 로드 시 더보기 버튼에 입력할 값
// 현재 몇개 표시되어있는지 계산해주어야 한다.
let count = getShownCounts();
moreCount.innerText = `(${count}/${finishedActivities.length})`;

moreBtn.addEventListener("click", () => {
    // 먼저 현재 몇개 표시되어있는지를 구한다.
    let currentShownCount = getShownCounts();
    // 남은 개수가 8개 이하라면 이번 클릭에서 더보기 버튼이 없어져야 한다.
    if (finishedActivities.length - currentShownCount <= 8) {
        moreBtn.style.display = "none";
        for (let i = currentShownCount - 1; i < finishedActivities.length; i++) {
            finishedActivities[i].style.display = "block";
        }
    }
    // 남은 개수가 9개 이상이라면 8개 더 표시 후에도 더보기 버튼이 남아있어야 한다.
    else {
        for (let i = currentShownCount - 1; i < currentShownCount + 8; i++) {
            finishedActivities[i].style.display = "block";
        }
        moreCount.innerText = `(${currentShownCount + 8}/${finishedActivities.length})`;
    }
});

// 탭 클릭 시 활성화되는 탭 변경,
// 그리고 아래 표시되는 내용 싹 다 변경
const activityFilterWrap = document.querySelector(".club-detail-filter-event");
const activityFilterBtn = document.querySelector(".club-detail-filter-event .club-detail-filter-button");
const infoFilterWrap = document.querySelector(".club-detail-filter-info");
const infoFilterBtn = document.querySelector(".club-detail-filter-info .club-detail-filter-button");
const tpFilterWrap = document.querySelector(".club-detail-filter-teenplay");
const tpFilterBtn = document.querySelector(".club-detail-filter-teenplay button");
const activityContent = document.querySelector("div.club-detail-desc-container");
const infoContent = document.querySelector(".club-info");
const tpContent = document.querySelector(".club-teenplay");
const noticeFilterWrap = document.querySelector(".club-detail-filter-notice");
const noticeFilterBtn = document.querySelector(".club-detail-filter-notice .club-detail-filter-button");
const noticeContent = document.querySelector(".club-notice");

activityFilterBtn.addEventListener("click", () => {
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
    tpContent.style.display = "none";
    noticeContent.style.display = "none";
});

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
    infoContent.style.display = "block";
    activityContent.style.display = "none";
    tpContent.style.display = "none";
    noticeContent.style.display = "none";
});

tpFilterBtn.addEventListener("click", () => {
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
    tpContent.style.display = "block";
    activityContent.style.display = "none";
    infoContent.style.display = "none";
    noticeContent.style.display = "none";
});

noticeFilterBtn.addEventListener("click", () => {
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
    noticeContent.style.display = "block";
    activityContent.style.display = "none";
    infoContent.style.display = "none";
    tpContent.style.display = "none";
});

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

// 모임 이름 받아와서 넣어야함(아래는 예시)
let clubName = "TEEN_PLAYERS";

// 가입신청 버튼 클릭 시 모달창 출력
const applyBtn = document.getElementById("apply");
applyBtn.addEventListener("click", () => {
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
            // 가입신청 관련 서버 작업 코드 입력
            Swal.fire("신청 완료", `[${clubName}] 모임에 가입 신청이 완료되었어요!`, "success");
        } else if (result.dismiss == "cancel") {
            return;
        }
    });
});

// 승인대기 버튼 클릭 시 신청취소 모달창 출력
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
            // 신청취소 관련 서버 작업 코드 입력
            Swal.fire("취소 완료", "가입 신청을 취소하였습니다.", "success");
        } else if (result.dismiss == "cancel") {
            return;
        }
    });
});

// 탈퇴하기 버튼 클릭 시 탈퇴하기 모달창 출력
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
            // 모임탈퇴 관련 서버 작업 코드 입력
            Swal.fire("모임 탈퇴", `[${clubName}] 모임에서 탈퇴하였습니다.`, "success");
        } else if (result.dismiss == "cancel") {
            return;
        }
    });
});

// 하트 아이콘 클릭 시 모달창 하트 이미지 변경, 모달창 출력
const activeLikeBtns = document.querySelectorAll(".club-detail-like-button");
const emptyHearts = document.querySelectorAll(".club-detail-like-button .club-detail-like-icon.empty");
const fullHearts = document.querySelectorAll(".club-detail-like-button .club-detail-like-icon.full");
// 전체 모달
const modalWrap = document.querySelector(".club-modal-wrap");
// 모달 중에서 관심 설정할 때 표시할 부분
const modalLikeContainer = document.querySelector(".club-modal-like-wrap:not(.unlike)");
// 모달 중에서 관심 해제할 때 표시할 부분
const modalUnlikeContainer = document.querySelector(".club-modal-like-wrap.unlike");

activeLikeBtns.forEach((button, i) => {
    button.addEventListener("click", () => {
        modalWrap.style.display = "block";
        if (emptyHearts[i].style.display === "none") {
            modalUnlikeContainer.style.display = "block";
            modalLikeContainer.style.display = "none";
            emptyHearts[i].style.display = "block";
            fullHearts[i].style.display = "none";
        } else {
            modalUnlikeContainer.style.display = "none";
            modalLikeContainer.style.display = "block";
            emptyHearts[i].style.display = "none";
            fullHearts[i].style.display = "block";
        }
    });
});

const finishedLikeBtns = document.querySelectorAll(".finished-events-like-btn");
const finishedEmptyHearts = document.querySelectorAll(".finished-events-like-btn .club-detail-like-icon.empty");
const finishedFullHearts = document.querySelectorAll(".finished-events-like-btn .club-detail-like-icon.full");
finishedLikeBtns.forEach((button, i) => {
    button.addEventListener("click", () => {
        modalWrap.style.display = "block";
        if (finishedEmptyHearts[i].style.display === "none") {
            modalUnlikeContainer.style.display = "block";
            modalLikeContainer.style.display = "none";
            finishedEmptyHearts[i].style.display = "block";
            finishedFullHearts[i].style.display = "none";
        } else {
            modalUnlikeContainer.style.display = "none";
            modalLikeContainer.style.display = "block";
            finishedEmptyHearts[i].style.display = "none";
            finishedFullHearts[i].style.display = "block";
        }
    });
});

// 모달 창 내 버튼 클릭 시 모달창 닫기
const modalLikeExitBtn = document.querySelector(".club-modal-like-button");
const modalUnlikeExitBtn = document.querySelector(".club-modal-unlike-button");

function exitModal() {
    modalWrap.style.display = "none";
}

modalLikeExitBtn.addEventListener("click", exitModal);
modalUnlikeExitBtn.addEventListener("click", exitModal);

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

// 틴플레이 업로드 버튼 클릭 시 모달창
const tpUploadModal = document.querySelector(".club-upload-modal-wrap");
const tpUploadBtn = document.querySelector(".club-detail-filter-upload-wrap");
const tpModalCloseBtn = document.querySelector(".upload-modal-close-wrap");
tpUploadBtn.addEventListener("click", () => {
    tpUploadModal.style.display = "block";
    tpUploadBtn.style.display = "none";
});
tpModalCloseBtn.addEventListener("click", () => {
    tpUploadModal.style.display = "none";
    tpUploadBtn.style.display = "flex";
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
thumbnailUploadBox.addEventListener("dragenter", (e) => {
    e.preventDefault();
});
thumbnailUploadBox.addEventListener("dragover", (e) => {
    e.preventDefault();
});
thumbnailUploadBox.addEventListener("dragleave", (e) => {
    e.preventDefault();
});
thumbnailUploadBox.addEventListener("drop", (e) => {
    e.preventDefault();
    let file = e.dataTransfer;
    let checkSize = 1024 * 1024 * THUMBNAIL_SIZE;
    if (!checkFileSize(file, checkSize)) {
        thumbnailSizeMsg.style.display = "block";
        e.preventDefault();
        return;
    }
    thumbnailSizeMsg.style.display = "none";
    fileSize = file.files[0].size;
    thumbnailSizeInfo.innerText = getFileSizeWithExtension(fileSize);
    thumbnailNameInfo.innerText = file.files[0].name;
    thumbnailUploadBox.classList.remove("appear");
    thumbnailUploadBox.classList.add("disappear");
    setTimeout(() => {
        thumbnailUploadBox.classList.add("hidden");
        uploadedThumbnailInfo.classList.remove("hidden");
        uploadedThumbnailInfo.classList.remove("disappear");
        uploadedThumbnailInfo.classList.add("appear");
    }, 501);
});

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

// 업로드 버튼 클릭 시 모달창으로 확인
finalSaveButton.addEventListener("click", () => {
    Swal.fire({
        title: "업로드하시겠습니까?",
        text: "한 번 업로드한 틴플레이는 수정이 불가능합니다.",
        icon: "warning",
        showCancelButton: true,
        confirmButtonColor: "#3085d6",
        cancelButtonColor: "#d33",
        confirmButtonText: "업로드",
        cancelButtonText: "취소",
    }).then((result) => {
        if (result.value) {
            // 틴플레이 업로드 관련 서버 작업 코드 입력
            tpModalCloseBtn.click();
            Swal.fire("업로드 진행중", "업로드를 진행합니다. <br> 업로드는 최대 5분 안에 완료됩니다!", "success");
        } else if (result.dismiss == "cancel") {
            return;
        }
    });
});

// 틴플레이 삭제 아이콘 마우스 올리면 색상 변경 및
// 틴플레이 삭제 아이콘 클릭 시 모달창 출력
// 실제 삭제는 서버에서 구현합니다.
const teenplayDeleteIconBlacks = document.querySelectorAll(".club-teenplay-delete");
const teenplayDeleteIconHovers = document.querySelectorAll(".club-teenplay-delete-hover");
const teenplayDeleteWraps = document.querySelectorAll(".club-teenplay-delete-wrap");
const teenplayContentsWrap = document.querySelector(".club-teenplay-contents-box");
const teenplayContents = document.querySelectorAll(".club-teenplay-contents");

teenplayDeleteWraps.forEach((div, i) => {
    div.addEventListener("mouseover", () => {
        teenplayDeleteIconBlacks[i].style.display = "none";
        teenplayDeleteIconHovers[i].style.display = "block";
    });
    div.addEventListener("mouseout", () => {
        teenplayDeleteIconBlacks[i].style.display = "block";
        teenplayDeleteIconHovers[i].style.display = "none";
    });
    div.addEventListener("click", () => {
        Swal.fire({
            title: "삭제하시겠습니까?",
            text: "삭제된 틴플레이는 복구가 불가능합니다.",
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#3085d6",
            cancelButtonColor: "#d33",
            confirmButtonText: "삭제",
            cancelButtonText: "취소",
        }).then((result) => {
            if (result.value) {
                // 틴플레이 삭제 관련 서버 작업 코드 입력
                // 완료 시 아래 코드 실행 (실제로는 또 .then(()=>{}) 으로 써야함)
                Swal.fire("삭제 완료", "틴플레이 삭제가 완료되었습니다.", "success");
                teenplayContentsWrap.removeChild(teenplayContents[i]);
            } else if ((result.dismiss = "cancel")) {
                return;
            }
        });
    });
});
