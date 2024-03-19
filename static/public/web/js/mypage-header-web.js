



const mypageServices = document.querySelector(".member-services");
const mypageMenu = document.querySelector(".mypage-menu");
const AllWhithoutClass = document.querySelectorAll("body :not(.member-services) :not(.mypage-menu)");

// 마이페이지 간이 목록 이외 클릭 시 none처리 이벤트
if (mypageServices && mypageMenu) {
    document.addEventListener("click", (e) => {
        if (!e.target.closest(".member-services")) {
            mypageMenu.classList.remove("display:flex");
        } else {
            console.log("else");
            if (e.target.closest(".member-service-menu-btn")) {
                mypageMenu.classList.toggle("display:flex");
            }
        }
    });
}
const searchInput = document.querySelector(".search-input");
// 헤더의 검색창 클릭 시 검색 모달창 block처리 이벤트
const searchModal = document.querySelector(".search-modal-container");
const searchButton = document.querySelector(".search-btn");

searchButton.addEventListener("click", () => {
    searchModal.style.display = "block";
    searchInput.focus();
});

// 검색창 클릭 시 검색 모달 block처리 이벤트
const backButton = document.querySelector(".back-btn");
const searchModalShadow = document.querySelector(".search-modal-shadow");

backButton.addEventListener("click", () => {
    searchModal.style.display = "none";
});

searchModalShadow.addEventListener("click", () => {
    searchModal.style.display = "none";
});

// 입력 시에 검사할 fetch 모듈
const getRealTimeSearchResult = async (keyword, callback) => {
    const response = await fetch(`/search/realtime-search/api?keyword=${keyword}`);
    const realTimeResults = await response.json();
    if (callback) {
        callback(keyword, realTimeResults);
    }
}

const addEmTagsToKeyword = (originalTitle, keyword) => {
    let regex = new RegExp("(" + keyword + ")", "gi");
    return originalTitle.replace(regex, "<em>$1</em>")
}

// 검색 입력 시 키워드에 해당하는 활동 및 모임 홍보 글 표기 이벤트

const searchActivitySection = document.querySelector(".search-activity-section");
const serchIconBtn = document.querySelector(".serch-icon-btn");
const resultEmptyMsg = document.querySelector(".search-realtime-empty");
const searchResultWrap = document.querySelector(".search-realtime-results");

const showRealTimeSearchResult = (keyword, realTimeResults) => {
    let text = ``;
    if (realTimeResults.length !== 0){
        resultEmptyMsg.style.display = "none";
        realTimeResults.forEach((activity) => {
            activity.activity_title = addEmTagsToKeyword(activity.activity_title, keyword);
            text += `
                <a class="activity-result" href="/activity/detail?id=${activity.id}">
                    ${activity.activity_title}
                </a>
            `
        })
        searchResultWrap.innerHTML = text;
        return;
    }
    resultEmptyMsg.style.display = "block";
    searchResultWrap.innerHTML = "";
}


searchInput.addEventListener("input", async (e) => {
    if (e.target.value) {
        searchActivitySection.style.display = "block";
        let keyword = e.target.value;
        await getRealTimeSearchResult(keyword, showRealTimeSearchResult);
        return;
    }
    searchActivitySection.style.display = "none";

});

// 검색 모듈
const recentSearchService = (() => {
    const getKeywords = async (callback) => {
        const response = await fetch(`/search/recent-keywords/api/`);
        const keywords = await response.json();
        if (callback) {
            callback(keywords);
        }
    }

    const deleteKeyword = async (keyword) => {
        await fetch(`/search/recent-keywords/api/`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify(keyword)
        });
    }

    const deleteAllKeywords = async () => {
        await fetch(`/search/recent-keywords-delete-all/api/`);
    }

    return {
        getKeywords: getKeywords,
        deleteKeyword: deleteKeyword,
        deleteAllKeywords: deleteAllKeywords
    }
})();


// 로그인 여부
const isLogin = document.querySelector("input[name=is-login]").value === "true";

// 로그인 여부에 따라 최근 검색어 표시 여부 결정
if (!isLogin) {
    document.querySelector(".recently-keyword-section").style.display = "none";
}

// 엔터키를 누를 경우 input의 value를 최근 검색기록 목록에 최신순으로 추가
const recentlyKeywordList = document.querySelector(".recently-keyword-list");

const showRecentKeywords = (keywords) => {
    let text = ``;
    keywords.forEach((keyword) => {
        text += `<form action="/activity/list/" method="post" class="keyword-click-form" style="cursor: pointer">`
        text += `<input type="hidden" name="csrfmiddlewaretoken" id="csrfmiddlewaretoken" value="${csrfToken}">`
        text += `<div class="recently-keyword-item">`;
        text += `<div class="search-keyword-link">${keyword.keyword}</div>`;
        text += `<button class="cancle-btn" type="button">`;
        text += `<svg xmlns="http://www.w3.org/2000/svg" class="cancle-svg" viewBox="0 0 20 20" fill="currentColor">`;
        text += `<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>`;
        text += `</svg>`;
        text += `</button>`;
        text += `<input type="hidden" class="recent-keyword-id" value="${keyword.id}">`
        text += `<input type="hidden" name="keyword" value="${keyword.keyword}">`
        text += `</div>`;
        text += `</form>`;
    })
    recentlyKeywordList.innerHTML = text;
    searchInput.value = "";
    countKeyword();
    createEvent();
    const keywordClickForms = document.querySelectorAll(".keyword-click-form");
    keywordClickForms.forEach((form) => {
        form.addEventListener("click", (e) => {
            form.submit();
        })
    })
}

if (isLogin) {
    // 최근검색어 목록에 뿌리기
    recentSearchService.getKeywords(showRecentKeywords)
}

searchInput.addEventListener("keyup", (e) => {
    let text = ``;
    if (e.keyCode === 13) {
        if (!e.target.value) {
            return;
        }
        // 검색 수행 후 검색 결과로 이동하는 코드
        document['list-form'].submit();
    }
});

// 최근 검색 기록 항목이 7개가 되면 7번째를 지우는 함수
const countKeyword = () => {
    const recentlyKeywordItem = document.querySelectorAll(".recently-keyword-item");

    if (recentlyKeywordItem.length > 6) {
        recentlyKeywordList.removeChild(recentlyKeywordItem[6]);
    }
};

// 검색어 삭제 버튼 클릭 시 해당 항목 삭제
const createEvent = () => {
    const cancleBtns = document.querySelectorAll(".cancle-btn");
    const keywordIds = document.querySelectorAll(".recent-keyword-id");
    cancleBtns.forEach((cancleBtn, i) => {
        cancleBtn.addEventListener("click", async (e) => {
            e.preventDefault();
            e.stopPropagation();
            cancleBtn.parentElement.remove();
            keyword = {
                id: keywordIds[i].value
            }
            await recentSearchService.deleteKeyword(keyword)
            await recentSearchService.getKeywords(showRecentKeywords);
        });
    });
};

createEvent();

// 검색 기록 삭제 클릭 시 최근 검색어 목록 지우기
const deleteSearchLogBtn = document.querySelector(".delete-search-log-btn");

deleteSearchLogBtn.addEventListener("click", async () => {
    await recentSearchService.deleteAllKeywords();
    await recentSearchService.getKeywords(showRecentKeywords);
});

// 검색 모달 내 더보기 버튼 클릭 시 더보기 사라지고 리스트에 클레스로 준 속성 지우기
const recommendKeywordsMore = document.querySelector(".recommend-keywords-more");
const recommendKeywordsList = document.querySelector(".recommend-keywords-list");

recommendKeywordsMore.addEventListener("click", (e) => {
    recommendKeywordsList.className = "";
    recommendKeywordsMore.style.display = "none";
});

// 닉네임 길이에 따라 헤더 내 알림 개수 위치 변경하기
const memberServiceWrap = document.querySelector(".member-services");
if (memberServiceWrap) {
    const memberNickname = document.querySelector("span.nicknaem").innerText;
    const alarmCountWrap = document.querySelector(".signal-sign-item");
    alarmCountWrap.style.left = `calc(100% - ${69 + (memberNickname.length - 2) * 6}px)`;


    // 알람 개수 띄우기
    const alarmCount1 = document.querySelector("div.signal-sign");
    const alarmCount2 = document.querySelector(".mypage-menu-signal-count");
    const alarmWrap1 = document.querySelector(".signal-sign-box");
    const alarmMemberId = document.querySelector("input[name=header-member-id]").value;

    const getAlarmCount = async (alarmMemberId, callback) => {
        const response = await fetch(`/member/alarms/api?member-id=${alarmMemberId}`);
        const alarmCount = await response.json();
        if (callback) {
            callback(alarmCount);
        }
    }

    const showAlarmCount = (alarmCount) => {
        if (Number(alarmCount) === 0) {
            alarmWrap1.style.display = "none";
            alarmCount2.style.display = "none";
            return;
        }
        alarmWrap1.style.display = "flex";
        alarmCount2.style.display = "flex";
        alarmCount1.innerText = Number(alarmCount) <= 99 ? alarmCount : '99+';
        alarmCount2.innerText = Number(alarmCount) <= 99 ? alarmCount : '99+';
    }

    getAlarmCount(alarmMemberId, showAlarmCount)

    // 카테고리 띄우기
    const headerCategoryWrap = document.querySelector(".category-group-items");

    const getCategories = async (callback) => {
        const response = await fetch(`/activity/categories/api/`);
        const categories = await response.json();
        if (callback) {
            callback(categories);
        }
    }

    const showCategories = (categories) => {
        let text = ``;
        categories.forEach((category) => {
            text += `
                <form class="category-form" action="/activity/list/" method="post" name="category-form${category.id}">
                    <input type="hidden" name="csrfmiddlewaretoken" id="csrfmiddlewaretoken" value="${csrfToken}">
                    <div class="category-item">
                        <div class="item-link">
                            <input type="hidden" name="category-id" value="${category.id}">
                            <span>${category.category_name}</span>
                        </div>
                    </div>
                </form>
            `;
        })
        headerCategoryWrap.innerHTML = text;
        const forms = document.querySelectorAll(".category-form");
        forms.forEach((form) => {
            form.addEventListener("click", (e) => {
                form.submit();
            })
        })
    }

    getCategories(showCategories);
}

// 추천 검색어의 각 검색어 클릭 시 검색 결과로 이동
const recommendKeywordForms = document.querySelectorAll(".recommend-keyword-form");
recommendKeywordForms.forEach((form) => {
    form.addEventListener("click", (e) => {
        form.submit();
    })
})




















