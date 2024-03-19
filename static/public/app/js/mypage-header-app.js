const sendEmailATag = document.querySelector(".send-email");
const sendEmailModal = document.querySelector(".send-email-modal-container");
const checkBtn = document.querySelector(".check-btn");

// 인증 메일 발송하기 클릭 시 모달창 block처리 이벤트
sendEmailATag.addEventListener("click", () => {
    sendEmailModal.style.display = "block";
});

// 모달창 내 확인 버튼 클릭 시 모달창 none처리 이벤트
checkBtn.addEventListener("click", () => {
    sendEmailModal.style.display = "none";
});

// 마이페이지 목록 버튼 클릭 시 발생 하는 이벤트
const memberServiceBtn = document.querySelector(".member-service-btn");
const mypageModalNav = document.querySelector(".mypage-modal-nav");
const mypageModalAside = document.querySelector(".mypage-modal-aside");

memberServiceBtn.addEventListener("click", () => {
    mypageModalAside.style.display = "flex";
});

//마이페이지 목록 내 X 버튼 클릭 시 발생하는 이벤트
const closeBtn = document.querySelector(".close-btn");

closeBtn.addEventListener("click", () => {
    mypageModalNav.style.animation = "slideOut 0.3s ease-in-out";
    setTimeout(() => {
        mypageModalNav.removeAttribute("style");
        mypageModalAside.style.display = "none";
    }, 300);
});

// 헤더의 검색창 클릭 시 검색 모달창 block처리 이벤트
const searchModal = document.querySelector(".search-modal-container");
const mypageSearchButton = document.querySelector(".mypage-search-btn");

mypageSearchButton.addEventListener("click", () => {
    searchModal.style.display = "block";
});

// 뒤로가기 클릭 시 검색 모달 none처리 이벤트
const backButton = document.querySelector(".back-btn");

backButton.addEventListener("click", () => {
    searchModal.style.display = "none";
});

// 검색 입력 시 키워드에 해당하는 활동 및 모임 홍보 글 표기 이벤트
const searchInput = document.querySelector(".search-input");
const searchActivitySection = document.querySelector(".search-activity-section");
const searchClupSection = document.querySelector(".search-clup-section");
const serchIconBtn = document.querySelector(".serch-icon-btn");

searchInput.addEventListener("input", (e) => {
    if (e.target.value) {
        searchActivitySection.style.display = "block";
        searchClupSection.style.display = "block";
        return;
    }
    searchActivitySection.style.display = "none";
    searchClupSection.style.display = "none";
});

// 엔터키를 누를 경우 input의 value를 최근 검색기록 목록에 최신순으로 추가
const recentlyKeywordAppList = document.querySelector(".recently-keyword-app-list");

searchInput.addEventListener("keyup", (e) => {
    let text = ``;
    if (e.keyCode === 13) {
        if (!e.target.value) {
            return;
        }
        text += `<div class="recently-keyword-item">`;
        text += `<a class="search-keyword-link" href="">${e.target.value}</a>`;
        text += `<button class="cancle-btn" type="button">`;
        text += `<svg xmlns="http://www.w3.org/2000/svg" class="cancle-svg" viewBox="0 0 20 20" fill="currentColor">`;
        text += `<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>`;
        text += `</svg>`;
        text += `</button>`;
        text += `</div>`;
        recentlyKeywordAppList.innerHTML = text + recentlyKeywordAppList.innerHTML;
        e.target.value = "";
        countKeyword();
        createEvent();
    }
});

// 최근 검색 기록 항목이 7개가 되면 7번째를 지우는 함수
const countKeyword = () => {
    const recentlyKeywordItem = document.querySelectorAll(".recently-keyword-item");

    if (recentlyKeywordItem.length > 6) {
        recentlyKeywordAppList.removeChild(recentlyKeywordItem[6]);
    }
};

// 검색어 삭제 버튼 클릭 시 해당 항목 삭제
const createEvent = () => {
    const cancleBtns = document.querySelectorAll(".cancle-btn");
    cancleBtns.forEach((cancleBtn) => {
        cancleBtn.addEventListener("click", (e) => {
            cancleBtn.parentElement.remove();
        });
    });
};

createEvent();

// 검색 기록 삭제 클릭 시 최근 검색어 목록 지우기
const deleteSearchLogBtn = document.querySelector(".delete-search-log-btn");

deleteSearchLogBtn.addEventListener("click", () => {
    recentlyKeywordAppList.innerHTML = "";
});

// 검색 모달 내 더보기 버튼 클릭 시 더보기 사라지고 리스트에 클레스로 준 속성 지우기
const recommendKeywordsMore = document.querySelector(".recommend-keywords-more");
const recommendKeywordsList = document.querySelector(".recommend-keywords-list");

recommendKeywordsMore.addEventListener("click", (e) => {
    recommendKeywordsList.className = "";
    recommendKeywordsMore.style.display = "none";
});
