// // 상태 변경 버튼
// const modalOpenButtons = document.querySelector(".member-user-list-button");
// // 취소 버튼
// const modalCloseButtons = document.querySelector(".admin-user-modal-left-button");
// const cancelButton = document.getElementById("modalCloseButton");
// // 적용하기 버튼
// const applyButton = document.getElementById("modalApplyButton");
//
// // 상태 변경 모달창
// const modal = document.getElementById("admin-user-modal");
// const modalBack = document.getElementById("admin-user-modal-backdrop");
//
// // 상태 변경 모달
// document.addEventListener("DOMContentLoaded", () => {
//
//     // 상태 변경 버튼 클릭 시 모달 오픈
//     modalOpenButtons.addEventListener("click", () => {
//         // 클릭된 버튼에 대해 사용자 ID를 가져와서 모달에 설정
//         const userId = this.getAttribute("data-target");
//         modal.setAttribute("data-user-id", userId);
//         // 모달 오픈
//         modal.classList.remove("hidden");
//         modalBack.classList.remove("hidden");
//     });
//
//     // 상태 변경 취소 버튼 클릭 시 모달 종료
//     modalCloseButtons.addEventListener("click", () => {
//         modal.classList.add("hidden");
//         modalBack.classList.add("hidden");
//     });
//
//     // 상태 변경 적용 버튼 클릭 시 변경 내용 적용 및 모달 종료
//     applyButton.addEventListener("click", () => {
//         const userId = modal.getAttribute("data-user-id");
//         const userStatus = document.querySelector(`[data-id="${user.id}"]`);
//
//         const checkedItems = document.querySelectorAll(
//             ".main-comment-list-checkbox:checked"
//         );
//
//         checkedItems.forEach((checkbox) => {
//             const targetId = checkbox.closest("li").getAttribute("data-number");
//             const targetUserStatus = document.querySelector(
//                 `[data-id="${targetId}"]`
//             );
//
//             // 여기에서 상태를 변경하는 로직을 추가
//             if (targetUserStatus.textContent.trim() === "활동중") {
//                 targetUserStatus.textContent = "정지";
//             } else if (targetUserStatus.textContent.trim() === "정지") {
//                 targetUserStatus.textContent = "활동중";
//             }
//         });
//
//         modal.classList.add("hidden");
//         modalBack.classList.add("hidden");
//     });
// });
//
// function resizeTextarea() {
//     const textarea = document.getElementById("modal-text");
//     textarea.style.height = "auto";
//     textarea.style.height = textarea.scrollHeight + "px";
// }
//
// // 활동중 / 정지 모달
//
// const modalDetailOpenButtons = document.querySelectorAll(".member-user-list-detail-button");
// const modalDetailCloseButtons = document.querySelectorAll(".admin-user-modal-left-detail-button");
// const modalAdddetailCloseButtons = document.querySelectorAll(".admin-user-modal-right-detail-button");
// const detailmodal = document.getElementById("admin-post-modal");
// const detailmodalBack = document.getElementById("admin-user-modal-backdrop");
//
// modalDetailOpenButtons.forEach((button) => {
//     button.addEventListener("click", (event) => {
//         const targetId = event.currentTarget.getAttribute("data-target");
//         currentTargetLi = document.querySelector(`li[data-id="${targetId}"]`);
//
//         // 모달 열기
//         detailmodal.classList.remove("hidden");
//         detailmodalBack.classList.remove("hidden");
//     });
// });
//
// modalDetailCloseButtons.forEach((button) => {
//     button.addEventListener("click", () => {
//         detailmodal.classList.add("hidden");
//         detailmodalBack.classList.add("hidden");
//     });
// });
// const confirmDeleteButtonss = document.querySelectorAll(
//     ".admin-user-modal-right-detail-button"
// );
//
// if (confirmDeleteButtonss.length > 0) {
//     confirmDeleteButtonss.forEach((button) => {
//         button.addEventListener("click", () => {
//             // 현재 대상 li 선택함
//             // if (currentTargetLi) {
//             // }
//
//             // 모달 닫기
//             detailmodal.classList.add("hidden");
//             detailmodalBack.classList.add("hidden");
//         });
//     });
// }

// 공개 / 비공개 버튼

const searchModal = document.getElementById("admin-message-modal-search");
const searchOpen = document.querySelector(".main-wish-sellect-button");
const searchSend = document.querySelector(".admin-message-modal-search-send");
const searchReceive = document.querySelector(".admin-message-modal-search-receive");
const searchText = document.querySelector(".main-wish-sellect-button-span");
const searchadd = document.querySelector(".admin-message-modal-search-donotreceive");
const svg = document.querySelector(".main-comment-info-button-svg");

// 검색 버튼 클릭 시 모달 열기
searchOpen.addEventListener("click", (event) => {
    // 이벤트 전파를 막기 위해 stopPropagation() 호출
    event.stopPropagation();
    svg.setAttribute("transform", "rotate(180)");
    searchModal.classList.remove("hidden");
});

// 모달 외부를 클릭했을 때 이벤트 처리
document.addEventListener("click", (event) => {
    if (event.target !== searchOpen && !searchModal.contains(event.target)) {
        // 클릭된 요소가 검색 버튼이 아니고 모달 창에 속하지 않으면 모달을 닫음
        svg.removeAttribute("transform");
        searchModal.classList.add("hidden");
    }
});

// "공개/비공개" 버튼 클릭 시 모달 닫고 텍스트 변경
searchReceive.addEventListener("click", () => {
    searchModal.classList.add("hidden");
    searchText.textContent = "전체";
    svg.removeAttribute("transform");
});
// "공개" 버튼 클릭 시 모달 닫고 텍스트 변경
searchSend.addEventListener("click", () => {
    searchModal.classList.add("hidden");
    searchText.textContent = "활동중";
    svg.removeAttribute("transform");
});

// "비공개" 버튼 클릭 시 모달 닫고 텍스트 변경

searchadd.addEventListener("click", () => {
    searchModal.classList.add("hidden");
    searchText.textContent = "정지";
    svg.removeAttribute("transform");
});

// -------------------------------------------------
// 공개 비공개 버튼
document.addEventListener("DOMContentLoaded", function () {
    const allItems = document.querySelectorAll(".main-user-list");

    function updateTotalCount() {
        // 숫자 세는 코드
        const visibleItems = document.querySelectorAll(
            ".main-user-list:not(.hidden)"
        );
        const totalCount = visibleItems.length;
        // 숫자를 표시할 요소에 totalCount를 업데이트합니다.
        document.getElementById("total-count").textContent = totalCount;
    }

    document
        .querySelector(".admin-message-modal-search-receive")
        .addEventListener("click", function () {
            allItems.forEach((item) => {
                item.classList.remove("hidden");
            });
            updateTotalCount(); // 버튼 클릭 후 숫자 업데이트
        });

    document
        .querySelector(".admin-message-modal-search-send")
        .addEventListener("click", function () {
            allItems.forEach((item) => {
                if (
                    item
                        .querySelector(".main-user-list-message")
                        .textContent.trim() !== "활동중"
                ) {
                    item.classList.add("hidden");
                } else {
                    item.classList.remove("hidden");
                }
            });
            updateTotalCount(); // 버튼 클릭 후 숫자 업데이트
        });

    document
        .querySelector(".admin-message-modal-search-donotreceive")
        .addEventListener("click", function () {
            allItems.forEach((item) => {
                if (
                    item
                        .querySelector(".main-user-list-message")
                        .textContent.trim() !== "정지"
                ) {
                    item.classList.add("hidden");
                } else {
                    item.classList.remove("hidden");
                }
            });
            updateTotalCount(); // 버튼 클릭 후 숫자 업데이트
        });

    // 페이지가 로드될 때도 숫자를 업데이트합니다.
    updateTotalCount();
});
//
// // 체크박스 채워주기
// document.addEventListener("DOMContentLoaded", function () {
//     const statusName = document.querySelector(".main-user-status-name");
//     const checkboxes = document.querySelectorAll(".main-comment-list-checkbox");
//
//     statusName.addEventListener("click", function () {
//         let allChecked = true;
//         checkboxes.forEach((checkbox) => {
//             if (!checkbox.checked) {
//                 allChecked = false;
//                 checkbox.checked = true;
//             }
//         });
//
//         if (allChecked) {
//             checkboxes.forEach((checkbox) => {
//                 checkbox.checked = false;
//             });
//         }
//     });
// });
