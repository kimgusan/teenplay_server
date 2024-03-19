

// document.addEventListener("DOMContentLoaded", () => {
    // 상태 변경 버튼 클릭 시 모달 오픈


//     // 상태 변경 적용 버튼 클릭 시 변경 내용 적용 및 모달 종료
//     applyButton.addEventListener("click", () => {
//         const userId = modal.getAttribute("data-id");
//         const userStatus = document.querySelector(`[data-id="${userId}"]`);
//
//         const checkedItems = document.querySelectorAll(".main-comment-list-checkbox:checked");
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


// // 분류 모달창
// const searchModal = document.getElementById("admin-message-modal-search");
// const searchOpen = document.querySelector(".main-wish-sellect-button");
// const searchSend = document.querySelector(".admin-message-modal-search-send");
// const searchReceive = document.querySelector(".admin-message-modal-search-receive");
// const searchText = document.querySelector(".main-wish-sellect-button-span");
// const searchadd = document.querySelector(".admin-message-modal-search-donotreceive");
// const svg = document.querySelector(".main-comment-info-button-svg");
//
// // 검색 버튼 클릭 시 모달 열기
// searchOpen.addEventListener("click", (event) => {
//     // 이벤트 전파를 막기 위해 stopPropagation() 호출
//     event.stopPropagation();
//     svg.setAttribute("transform", "rotate(180)");
//     searchModal.classList.remove("hidden");
// });
//
// // 모달 외부를 클릭했을 때 이벤트 처리
// document.addEventListener("click", (event) => {
//     if (event.target !== searchOpen && !searchModal.contains(event.target)) {
//         // 클릭된 요소가 검색 버튼이 아니고 모달 창에 속하지 않으면 모달을 닫음
//         svg.removeAttribute("transform");
//         searchModal.classList.add("hidden");
//     }
// });
//
// // "공개/비공개" 버튼 클릭 시 모달 닫고 텍스트 변경
// searchReceive.addEventListener("click", () => {
//     searchModal.classList.add("hidden");
//     searchText.textContent = "전체";
//     svg.removeAttribute("transform");
// });
// // "공개" 버튼 클릭 시 모달 닫고 텍스트 변경
// searchSend.addEventListener("click", () => {
//     searchModal.classList.add("hidden");
//     searchText.textContent = "활동중";
//     svg.removeAttribute("transform");
// });
//
// // "비공개" 버튼 클릭 시 모달 닫고 텍스트 변경
// searchadd.addEventListener("click", () => {
//     searchModal.classList.add("hidden");
//     searchText.textContent = "정지";
//     svg.removeAttribute("transform");
// });
