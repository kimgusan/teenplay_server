document.addEventListener("DOMContentLoaded", function () {
    const modalOpenButtons = document.querySelectorAll(
        ".member-user-list-button"
    );
    const modalCloseButton = document.querySelector(
        ".admin-user-modal-left-button"
    );
    const modal = document.getElementById("admin-user-modal");
    const modalBack = document.getElementById("admin-user-modal-backdrop");
    const applyButton = document.getElementById("modalApplyButton");
    const cancelButton = document.getElementById("modalCloseButton");

    modalOpenButtons.forEach((button) => {
        button.addEventListener("click", function () {
            // 클릭된 버튼에 대해 사용자 ID를 가져와서 모달에 설정
            const userId = this.getAttribute("data-target");
            modal.setAttribute("data-user-id", userId);
            modal.classList.remove("hidden");
            modalBack.classList.remove("hidden");
        });
    });

    modalCloseButton.addEventListener("click", function () {
        modal.classList.add("hidden");
        modalBack.classList.add("hidden");
    });

    cancelButton.addEventListener("click", function () {
        modal.classList.add("hidden");
        modalBack.classList.add("hidden");
    });

    applyButton.addEventListener("click", function () {
        const userId = modal.getAttribute("data-user-id");
        const userStatus = document.querySelector(`[data-id="${userId}"]`);

        const checkedItems = document.querySelectorAll(
            ".main-comment-list-checkbox:checked"
        );

        checkedItems.forEach((checkbox) => {
            const targetId = checkbox.closest("li").getAttribute("data-number");
            const targetUserStatus = document.querySelector(
                `[data-id="${targetId}"]`
            );

            // 여기에서 상태를 변경하는 로직을 추가
            if (targetUserStatus.textContent.trim() === "활동중") {
                targetUserStatus.textContent = "정지";
            } else if (targetUserStatus.textContent.trim() === "정지") {
                targetUserStatus.textContent = "활동중";
            }
        });

        modal.classList.add("hidden");
        modalBack.classList.add("hidden");
    });
});

function resizeTextarea() {
    const textarea = document.getElementById("modal-text");
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + "px";
}

// ----------------------------------------------
// 댓글 모달창

// 회원모달창
const modalMessageOpenButtons = document.querySelectorAll(
    ".member-comment-list-button"
);
const modaMessagelCloseButtons = document.querySelectorAll(
    ".admin-comment-modal-left-button"
);
const modalMessageAddCloseButtons = document.querySelectorAll(
    ".admin-comment-modal-right-button"
);

const Messagemodal = document.getElementById("admin-comment-modal");
const MessagemodalBack = document.getElementById(
    "admin-comment-modal-backdrop"
);

modalMessageOpenButtons.forEach((button) => {
    button.addEventListener("click", () => {
        Messagemodal.classList.remove("hidden");
        MessagemodalBack.classList.remove("hidden");
    });
});

modaMessagelCloseButtons.forEach((button) => {
    button.addEventListener("click", () => {
        Messagemodal.classList.add("hidden");
        MessagemodalBack.classList.add("hidden");
    });
});

modalMessageAddCloseButtons.forEach((button) => {
    button.addEventListener("click", () => {
        Messagemodal.classList.add("hidden");
        MessagemodalBack.classList.add("hidden");
    });
});

// 검색창 모달

const searchModal = document.getElementById("admin-comment-modal-search");
const searchOpen = document.querySelector(".main-comment-info-button");
const searchSend = document.querySelector(".admin-comment-modal-search-send");
const searchReceive = document.querySelector(
    ".admin-comment-modal-search-receive"
);

const searchText = document.querySelector(".main-comment-info-button-text");

const path = document.querySelector(".main-comment-info-button-svg");

// 검색 버튼 클릭 시 모달 열기
searchOpen.addEventListener("click", (event) => {
    // 이벤트 전파를 막기 위해 stopPropagation() 호출
    event.stopPropagation();
    searchModal.classList.remove("hidden");
    path.setAttribute("transform", "rotate(180)");
});

// 모달 외부를 클릭했을 때 이벤트 처리
document.addEventListener("click", (event) => {
    if (event.target !== searchOpen && !searchModal.contains(event.target)) {
        // 클릭된 요소가 검색 버튼이 아니고 모달 창에 속하지 않으면 모달을 닫음
        path.removeAttribute("transform");
        searchModal.classList.add("hidden");
    }
});

// "작성자" 버튼 클릭 시 모달 닫고 텍스트 변경
searchSend.addEventListener("click", () => {
    searchModal.classList.add("hidden");
    path.removeAttribute("transform");
    if (searchText.textContent === "게시글") {
        searchText.textContent = "작성자";
    }
});

// "게시글" 버튼 클릭 시 모달 닫고 텍스트 변경
searchReceive.addEventListener("click", () => {
    searchModal.classList.add("hidden");
    path.removeAttribute("transform");
    if (searchText.textContent === "작성자") {
        searchText.textContent = "게시글";
    }
});

// 리스트 삭제

const modalDeleteOpenButtons = document.querySelectorAll(
    ".member-user-list-delete-button"
);
const modalDeleteCloseButtons = document.querySelectorAll(
    ".admin-user-modal-left-delete-button"
);
const modalDeleteAddCloseButtons = document.querySelectorAll(
    ".admin-user-modal-right-delete-button"
);

const deletemodal = document.getElementById("admin-user-modal-delete");
const deletemodalBack = document.getElementById("admin-comment-modal-backdrop");
let currentTargetLi;

modalDeleteCloseButtons.forEach((button) => {
    button.addEventListener("click", () => {
        deletemodal.classList.add("hidden");
        deletemodalBack.classList.add("hidden");
    });
});

modalDeleteOpenButtons.forEach((button) => {
    button.addEventListener("click", (event) => {
        const targetId = event.currentTarget.getAttribute("data-target");
        currentTargetLi = document.querySelector(
            `li[data-number="${targetId}"]`
        );

        // 모달 열기
        deletemodal.classList.remove("hidden");
        deletemodalBack.classList.remove("hidden");
    });
});

const deleteButton = document.querySelector(
    ".admin-user-modal-right-delete-button"
);

deleteButton.addEventListener("click", () => {
    const checkedItems = document.querySelectorAll(
        ".main-comment-list-checkbox:checked"
    );

    checkedItems.forEach((checkbox) => {
        const targetId = checkbox.closest("li").getAttribute("data-number");
        const targetLi = document.querySelector(
            `li[data-number="${targetId}"]`
        );
        if (targetLi) {
            targetLi.remove();
        }
    });

    updateTotalCount();

    // 모달 닫기
    deletemodal.classList.add("hidden");
    deletemodalBack.classList.add("hidden");
});

function updateTotalCount() {
    // 각종 업데이트 코드
}

// 여기서부터 활동중 / 정지 모달 관리

const addsearchModal = document.getElementById(
    "admin-message-modal-search-add"
);
const addsearchOpen = document.querySelector(".main-wish-sellect-button-add");
const addsearchSend = document.querySelector(
    ".admin-message-modal-search-send-add"
);
const addsearchReceive = document.querySelector(
    ".admin-message-modal-search-receive-add"
);
const addsearchText = document.querySelector(
    ".main-wish-sellect-button-span-add"
);
const addsearchadd = document.querySelector(
    ".admin-message-modal-search-donotreceive-add"
);

const addsvg = document.querySelector(".main-comment-info-button-svga");
// 검색 버튼 클릭 시 모달 열기
addsearchOpen.addEventListener("click", (event) => {
    // 이벤트 전파를 막기 위해 stopPropagation() 호출
    event.stopPropagation();
    addsvg.setAttribute("transform", "rotate(180)");
    addsearchModal.classList.remove("hidden");
});

// 모달 외부를 클릭했을 때 이벤트 처리
document.addEventListener("click", (event) => {
    if (
        event.target !== addsearchOpen &&
        !addsearchModal.contains(event.target)
    ) {
        // 클릭된 요소가 검색 버튼이 아니고 모달 창에 속하지 않으면 모달을 닫음
        addsearchModal.classList.add("hidden");
        addsvg.removeAttribute("transform");
    }
});

// "공개/비공개" 버튼 클릭 시 모달 닫고 텍스트 변경
addsearchReceive.addEventListener("click", () => {
    addsearchModal.classList.add("hidden");
    addsearchText.textContent = "전체";
    addsvg.removeAttribute("transform");
});
// "공개" 버튼 클릭 시 모달 닫고 텍스트 변경
addsearchSend.addEventListener("click", () => {
    addsearchModal.classList.add("hidden");
    addsearchText.textContent = "활동중";
    addsvg.removeAttribute("transform");
});

// "비공개" 버튼 클릭 시 모달 닫고 텍스트 변경

addsearchadd.addEventListener("click", () => {
    addsearchModal.classList.add("hidden");
    addsearchText.textContent = "정지";
    addsvg.removeAttribute("transform");
});

// 공개 비공개 버튼 각 찾아서 하는거
document.addEventListener("DOMContentLoaded", function () {
    const allItems = document.querySelectorAll(".main-comment-list");

    function updateTotalCount() {
        // 숫자 세는 코드
        const visibleItems = document.querySelectorAll(
            ".main-comment-list:not(.hidden)"
        );
        const totalCount = visibleItems.length;
        // 숫자를 표시할 요소에 totalCount를 업데이트합니다.
        document.getElementById("total-count").textContent = totalCount;
    }

    document
        .querySelector(".admin-message-modal-search-receive-add")
        .addEventListener("click", function () {
            allItems.forEach((item) => {
                item.classList.remove("hidden");
                updateTotalCount();
            });
            updateTotalCount();
        });
    updateTotalCount();

    document
        .querySelector(".admin-message-modal-search-send-add")
        .addEventListener("click", function () {
            allItems.forEach((item) => {
                if (
                    item
                        .querySelector(".main-comment-list-paycount")
                        .textContent.trim() !== "활동중"
                ) {
                    item.classList.add("hidden");
                } else {
                    item.classList.remove("hidden");
                }
            });
            updateTotalCount();
        });

    document
        .querySelector(".admin-message-modal-search-donotreceive-add")
        .addEventListener("click", function () {
            allItems.forEach((item) => {
                if (
                    item
                        .querySelector(".main-comment-list-paycount")
                        .textContent.trim() !== "정지"
                ) {
                    item.classList.add("hidden");
                } else {
                    item.classList.remove("hidden");
                }
            });
            updateTotalCount();
        });
    updateTotalCount();
    // 페이지가 로드될 때도 숫자를 업데이트합니다.
});

// 체크박스 채워주기
document.addEventListener("DOMContentLoaded", function () {
    const statusName = document.querySelector(".main-comment-status-name");
    const checkboxes = document.querySelectorAll(".main-comment-list-checkbox");

    statusName.addEventListener("click", function () {
        let allChecked = true;
        checkboxes.forEach((checkbox) => {
            if (!checkbox.checked) {
                allChecked = false;
                checkbox.checked = true;
            }
        });

        if (allChecked) {
            checkboxes.forEach((checkbox) => {
                checkbox.checked = false;
            });
        }
    });
});
