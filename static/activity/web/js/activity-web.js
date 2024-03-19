// 활동 표시 필터 및 페이지 초기화
let page = 1;
let date = "모든날";
let region = '';
let categories = [];

if (selectedCategory !== 'None' && selectedCategory) {
    categories.push(selectedCategory);
}
let showFinished = false;
let ordering = '새 행사순';
let searchKeyword = document.querySelector("input[name=search-keyword]").value;

const getList = async (keyword, page, date, region, categories, showFinished, ordering, callback) => {
    const response = await fetch(`/activity/lists/api/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json;charset=utf-8',
            'X-CSRFToken': csrf_token
        },
        body: JSON.stringify({
            keyword: keyword,
            page: page,
            date: date,
            region: region,
            categories: categories,
            showFinished: showFinished,
            ordering: ordering
        })
    });
    const activityLists = await response.json()
    if (callback) {
        callback(activityLists);
    }
}

const foundActivityCountBox = document.querySelector(".text-title-primary")
const emptyActivityMessage = document.querySelector(".filter-display-none")
const activityContentWrap = document.querySelector(".activity-find-all")
const pageWrap = document.querySelector(".page-box-items");
const makeDatetime = (date) => {
    const getDayOfWeek = (date) => {
        const days = ['일', '월', '화', '수', '목', '금', '토'];
        return days[date.getDay()];
    }
    const datetime = new Date(date);
    return `${datetime.getMonth() + 1}월${datetime.getDate()}일(${getDayOfWeek(datetime)})`;
}

const addClickEventToPages = (pageInfo) => {
    const leftArrowButton = document.querySelector(".left-move-button-bg");
    const rightArrowButton = document.querySelector(".right-move-button-bg");
    const pageButtons = document.querySelectorAll(".page-buttons");
    const pageTexts = document.querySelectorAll(".page-number-f");

    leftArrowButton.addEventListener("click", async () => {
        if (pageInfo.currentPage <= 1) return;
        page--;
        await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
    })

    rightArrowButton.addEventListener("click", async () => {
        if (pageInfo.currentPage === pageInfo.realEnd) return;
        page++;
        await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
    })

    pageButtons.forEach((btn, i) => {
        btn.addEventListener("click", async (e) => {
            page = pageTexts[i].innerText;
            await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
        })
    })
}

const updatePageLists = (pageInfo) => {
    const totalCount = pageInfo.totalCount;
    const startPage = pageInfo.startPage;
    const endPage = pageInfo.endPage;
    const currentPage = pageInfo.page;
    const realEnd = pageInfo.realEnd;
    const pageCount = pageInfo.pageCount;
    let pageText = ``;

    pageText = `
        <button class="left-move-button-bg ${currentPage === 1 ? 'disabled' : ''}" type="button">
            <svg class="left-move-icon" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clip-rule="evenodd"></path>
            </svg>
        </button>
        <div class="page-move-list">
    `;
    for (let i=startPage; i <= endPage; i++){
        pageText += `
            <button class="page-buttons ${currentPage === i ? 'page-number-f-btn' : 'page-number-btn'}" type="button">
                <div class="page-number-f">${i}</div>
            </button>
        `
    }
    pageText += `
        </div>
        <button class="right-move-button-bg ${currentPage === realEnd ? 'disabled' : ''}" type="button">
            <svg class="left-move-icon" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd"></path>
            </svg>
        </button>
    `
    pageWrap.innerHTML = pageText;
    addClickEventToPages(pageInfo);
}


// 활동마다 하트 클릭 시 관심 활동 등록/삭제 적용되도록

const addClickEventManageLike = () => {
    // 좋아요 선택 시 동작하는 js
    const activeLikeBtnsCopy = document.querySelectorAll(".like-btn-shadow");
    const emptyHeartsCopy = document.querySelectorAll(".like-btn-shadow .like-btn-display");
    const fullHeartsCopy = document.querySelectorAll(".like-btn-shadow .like-btn-display-none");
    const modalWrapCopy = document.querySelector(".club-modal-wrap");
    // > 좋아요 선택 시 관심 설정할 때 표시할 부분
    const modalLikeContainerCopy = document.querySelector(".club-modal-like-wrap:not(.unlike)");
    // > 좋아요 선택 시 관심 해제할 때 표시할 부분
    const modalUnlikeContainerCopy = document.querySelector(".club-modal-like-wrap.unlike");

    activeLikeBtnsCopy.forEach((button, i) => {
        button.addEventListener("click", async () => {
            modalWrapCopy.style.display = "block";
            if (emptyHeartsCopy[i].style.display === "none") {
                await activityLikeCountService.addOrDeleteLike(emptyHeartsCopy[i].classList[1], false);
                modalUnlikeContainerCopy.style.display = "block";
                modalLikeContainerCopy.style.display = "none";
                emptyHeartsCopy[i].style.display = "block";
                fullHeartsCopy[i].style.display = "none";
            } else {
                await activityLikeCountService.addOrDeleteLike(fullHeartsCopy[i].classList[1], true);
                modalUnlikeContainerCopy.style.display = "none";
                modalLikeContainerCopy.style.display = "block";
                emptyHeartsCopy[i].style.display = "none";
                fullHeartsCopy[i].style.display = "block";
            }
        });
    });

    // 좋아요 선택 시 창 내 버튼 클릭 시 모달창 닫기
    const modalLikeExitBtnCopy = document.querySelector(".club-modal-like-button");
    const modalUnlikeExitBtnCopy = document.querySelector(".club-modal-unlike-button");

    function exitModal() {
        modalWrapCopy.style.display = "none";
    }

    modalLikeExitBtnCopy.addEventListener("click", exitModal);
    modalUnlikeExitBtnCopy.addEventListener("click", exitModal);
}

const showList = (activityLists) => {
    foundActivityCountBox.innerText = activityLists[activityLists.length-1]
    let text = ``;
    if (activityLists.length === 2) {
        activityLists.pop();
        emptyActivityMessage.style.display = "block";
        const pageInfo = activityLists.pop()
        updatePageLists(pageInfo);
        activityContentWrap.innerHTML = text;
        addClickEventManageLike();
    } else {
        activityLists.pop();
        emptyActivityMessage.style.display = "none";
        const pageInfo = activityLists.pop()
        updatePageLists(pageInfo);

        activityLists.forEach((activity) => {
            let isLike = activity.is_like;
            text += `
                <div class="activity-window-bg">
                    <div class="activity-window">
                        <div class="act-window-image">
                            <a class="act-image-block" href="/activity/detail?id=${activity.id}">
                                <img class="act-image" src="/upload/${activity.thumbnail_path}" alt="" />
                            </a>
                            <div class="like-button-box">
                                <button class="like-btn-shadow" type="button">
                                    <span class="like-btn-display-none ${activity.id}" style="display: ${isLike ? 'block' : 'none'}">
                                        <svg class="like-btn" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                            <path fill-rule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clip-rule="evenodd"></path>
                                        </svg>
                                    </span>
                                    <span class="like-btn-display ${activity.id}" style="display: ${isLike ? 'none' : 'block'}">
                                        <svg class="like-btn-two" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path>
                                        </svg>
                                    </span>
                                </button>
                            </div>
                        </div>
                        <div class="act-window-content">
                            <div class="act-content-date-location">
                                <div class="act-content-date">
                                    <span class="act-content-date-text-inline">${makeDatetime(activity.activity_start)}</span>
                                </div>
                            </div>
                            <div class="act-content-title">
                                <a class="move-content" href="/activity/detail?id=${activity.id}">
                                    ${activity.activity_title}
                                </a>
                            </div>
                            <div class="act-content-money">
                                <div class="check-view">
                                    <span class="check-view-text">참가 ${activity.member_count}명</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        })
        activityContentWrap.innerHTML = text;
        addClickEventManageLike();
    }
}

getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);

// (일시) 라디오 버튼을 모두 돌면서 1개만 선택하도록 체크
document.querySelectorAll(".content-date .radio-everyday").forEach(function (radio) {
    radio.addEventListener("change", function () {
        // > 모든 라디오 버튼을 돌면서 현재 선택된 것 이외의 것들을 언체크
        document.querySelectorAll(".content-date .radio-everyday").forEach(function (otherRadio) {
            if (otherRadio !== radio) {
                otherRadio.checked = false;
                otherRadio.parentNode.nextElementSibling.style.display = "none"; // 다른 라디오 버튼의 텍스트
            }
        });
    });
});


//  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// 기간선택 체크 시 기간 달력 display 유무 표시
const checkedDate = document.getElementById("ev-radio-102");

const dateRangeCheck = document.querySelectorAll(".date-container .date-box .radio-everyday");
dateRangeCheck.forEach((selectDateRange) => {
    selectDateRange.addEventListener("click", (e) => {
        const dateRange = document.querySelector(".date-display-none");
        if (selectDateRange.id === "ev-radio-102" && selectDateRange.checked === true) {
            dateRange.style.display = "block";
        } else {
            dateRange.style.display = "none";
        }
    });
});

//  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



// 추천순을 선택 했을 때 bold 부분
const buttons = document.querySelectorAll("button[name='selected-recommend-btn']");
const filterSpans = document.querySelectorAll("span.filter-span");

buttons.forEach((button, i) => {
    button.addEventListener("click", () => {
        filterSpans.forEach(async (span, j) => {
            if (i === j && !span.classList.contains("font-bold-choice")) {
                span.classList.add("font-bold-choice");
                ordering = span.innerText;
                page = 1;
                await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
            } else if (i !== j && span.classList.contains("font-bold-choice")) {
                span.classList.remove("font-bold-choice");
            }
        });
    });
});

// 검색해서 들어왔을 시 필터 추가해놓기
if (searchKeyword) {
    const keywordFilterWrap = document.querySelector(".find-filter-keyword");
    const keywordFilterContent = document.getElementById("keyword-text");
    keywordFilterContent.innerText = `검색어: ${searchKeyword}`;
    keywordFilterWrap.style.display = "flex";
}

//  radio 버튼 클릭 된 값을 가져와서 hidden box의 명칭을 변경하기
let radioCheck = document.querySelectorAll(".date-box .radio-everyday");

radioCheck.forEach((radio) => {
    radio.addEventListener("click", async (e) => {
        // 클릭한 위치의 value 값을 div의 text 명칭으로 변경하기
        let dateTextRadio = document.querySelector(".find-filter-bg");
        let dateText = document.getElementById("date-text");
        if (e.target.value === "모든날") {
            dateText.innerHTML = e.target.value;
            dateTextRadio.style.display = "none";
        } else {
            dateText.innerHTML = e.target.value;
            dateTextRadio.style.display = "flex";
        }
        if (e.target.value === "기간선택"){
            return;
        }
        date = e.target.value;
        page = 1;
        await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
    });
});

// 지역
let blockLocation = document.querySelector(".block-location");
blockLocation.addEventListener("change",async (e) => {
    let locationTextRadio = document.querySelector(".find-filter-bg-location");
    let locationText = document.getElementById("location-content");
    if (e.target.value === "전체") {
        locationText.innerHTML = e.target.value;
        locationTextRadio.style.display = "none";
        region = '';
        page = 1;
        await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
    } else {
        locationText.innerHTML = e.target.value;
        locationTextRadio.style.display = "flex";
        region = e.target.value;
        page = 1;
        await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
    }
});

// 체크박스를 선택했을 때 div 추가되는 구문
let endDateTie = document.querySelector(".end-date-tie");
endDateTie.addEventListener("click", async (e) => {
    if (e.target.tagName === "INPUT") {
        if (e.target.checked) {
            let hiddenBg = document.querySelector(".hidden-filter-container-check");
            hiddenBg.innerHTML += `<div class="find-filter-bg-two" id="date-end-activity" style="display: flex" value="${e.target.value}">
                                        <div id="date-text">${e.target.value}</div>
                                        <button class="find-filter-exit">
                                            <svg class="exit-icon-button" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" id="exit-icon-button">
                                                <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                                            </svg>
                                        </button>
                                    </div>`;
            showFinished = true;
            page = 1;
            await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
        }
    }
});

// 반대로 해당 버튼을 클릭했을 때 체크가 풀리는 부분을 만들어 줌
endDateTie.addEventListener("click", async (e) => {
    if (e.target.tagName === "INPUT") {
        if (!e.target.checked) {
            let values = e.target.value;
            let divToRemove = document.querySelector(".find-filter-bg-two[value='" + values + "']");
            if (divToRemove) {
                divToRemove.remove();
            }
            showFinished = false;
            page = 1;
            await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
            // divToRemove.parentNode.removeChild(divToRemove);
            // 위에서 div 생성 시 id값으로 생성해 줬기 때문에 해당 id 값을 찾아서 div를 지워주면 된다
        }
    }
});

// 버튼을 클릭했을 때 활동분야 display(checkbox) 선택해서 보여주기 위한 div 추가
// > 활동분야의 체크박스를 선택했을 떄 나오는 input.value값을 알기위한 변수
let actCategoryCenter = document.querySelectorAll(".activity-center");

// > 활동분야 내부에 여러 카테고리 중 클릭이 가능한 부분이 있는지 확인을 위한 반복 구문 사용
let categoryCheck = new Array(actCategoryCenter.length);
categoryCheck.fill(false);

// 각 활동분야 선택 시 hiddenBg에 내용 띄우기 (필터 추가하기)
const manageCategoryHiddenBg = async (checkbox, i) => {
    const hiddenBg = document.querySelector(".hidden-filter-container-check");
    if (!categoryCheck[i]){
        hiddenBg.innerHTML += `<div class="find-filter-bg-three ${checkbox.nextElementSibling.innerText} ${checkbox.value}" id="checkbox${checkbox.value}" style="display: flex">
                                    <div id="date-text">${checkbox.nextElementSibling.innerText}</div>
                                    <button class="find-filter-exit">
                                        <svg class="exit-icon-button" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" id="exit-icon-button">
                                            <path class="exit-icon-button" fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                                        </svg>
                                    </button>
                                </div>`;
        categoryCheck[i] = true;
        if (!categories.includes(checkbox.value)){
            categories.push(checkbox.value);
        }
        page = 1;
        await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
        return;
    }
    const filterToRemove = document.getElementById(`checkbox${checkbox.value}`);
    filterToRemove.remove();
    categoryCheck[i] = false;
    if (categories.includes(checkbox.value)){
        categories = categories.filter(item => item !== checkbox.value);
    }
    page = 1;
    await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
}

actCategoryCenter.forEach( (center, i) => {
    const checkbox = center.firstElementChild;
    checkbox.addEventListener("click", async () => {
        await manageCategoryHiddenBg(checkbox, i);
    })
    if (categories.includes(checkbox.value)) {
        checkbox.click();
    }
})

//  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// 필터 타이틀에서 x 버튼 클릭 시 display 가 none으로 변경되고 초기 상태로 돌아가는 기능
let checkExit = document.getElementById("hidden-filter-container-check");

// 검색 키워드 있을 시 x를 클릭하여 검색 필터 초기화
checkExit.addEventListener("click", async (e) => {
    if (e.target.classList.contains("exit-icon-button")) {
        let parentDiv = e.target.closest(".find-filter-keyword");
        if (parentDiv) {
            parentDiv.style.display = "none";
            document.getElementById("keyword-text").innerText = "";
            searchKeyword = "";
            page = 1;
            await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
        }
    }
})

checkExit.addEventListener("click", (e) => {
    if (e.target.classList.contains("exit-icon-button")) {
        let parentDiv = e.target.closest(".find-filter-bg");
        if (parentDiv) {
            parentDiv.style.display = "none";
            document.querySelectorAll(".content-date .radio-everyday").forEach(async function (otherRadio) {
                if (otherRadio.id === "ev-radio-98") {
                    otherRadio.checked = true;
                    date = '모든날';
                    page = 1;
                    const dateRange = document.querySelector(".date-display-none");
                    dateRange.style.display = "none";
                    await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
                } else {
                    otherRadio.checked = false;
                }
            });
        }
    }
});

// 지역
checkExit.addEventListener("click", async (e) => {
    if (e.target.classList.contains("exit-icon-button")) {
        let parentDiv = e.target.closest(".find-filter-bg-location");
        if (parentDiv) {
            parentDiv.style.display = "none";
            // 셀렉트 박스 요소를 가져옵니다.
            const selectElement = document.querySelector(".block-location");

            // 첫 번째 옵션을 기본(default) 값으로 선택합니다.
            selectElement.selectedIndex = 0;

            region = '';
            page = 1;
            await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
        }
    }
});

// 분야
checkExit.addEventListener("click", async (e) => {
    if (e.target.classList.contains("exit-icon-button")) {
        let parentDiv = e.target.closest(".find-filter-bg-three");
        if (parentDiv) {
            const checkbox = document.querySelector(`.activity-chk.checkbox${parentDiv.classList[2]}`);
            checkbox.checked = false;
            categoryCheck[Number(parentDiv.classList[2])-1] = false;
            categories = categories.filter(item => item !== parentDiv.classList[2]);
            parentDiv.remove();
            page = 1;
            await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
        }
    }
});

// 모집 종료 행사
let endDateCheck = document.querySelectorAll(".filter-space .end-date-tie");
endDateCheck.forEach((textCategoryValue) => {
    document.addEventListener("click", async (e) => {
        let divTest = e.target.closest("div");
        var checkedValue = divTest.getAttribute("value");
        // check가 되어있는 것 중에 값이 값이 같은게 있으면 삭제
        let chkBtn = document.querySelectorAll(".filter-space .end-date-tie .end-date-chk");
        if (textCategoryValue.innerText === checkedValue) {
            chkBtn.forEach((inputName) => {
                let divToRemove = document.querySelector(".find-filter-bg-two[value='" + checkedValue + "']");
                if (divToRemove) {
                    divToRemove.remove();
                }
                if (inputName.getAttribute("value") === checkedValue) {
                    inputName.checked = false;
                }
            });
            page = 1;
            showFinished = false;
            await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
        }
    });
});


//  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// 핕터 초기화
let filterReset = document.querySelector(".filter-reset");
filterReset.addEventListener("click", async (e) => {
    // 달력 초기화
    dateRange = document.querySelector(".date-display-none");
    dateRange.style.display = "none";

    // 일시 초기화
    let dateTextRadio = document.querySelector(".find-filter-bg");
    let dateText = document.getElementById("date-text");
    dateTextRadio.style.display = "none";
    dateText.innerHTML = "모든날";
    document.querySelectorAll(".content-date .radio-everyday").forEach(function (otherRadio) {
        if (otherRadio.id === "ev-radio-98") {
            otherRadio.checked = true;
        } else {
            otherRadio.checked = false;
        }
    });

    // 모집 종료 행사 필터에 선택되어 보여지는 태그 삭제
    let divToJoinRemove = document.querySelectorAll(".find-filter-bg-two");
    divToJoinRemove.forEach((names) => {
        names.remove();
    });

    //  모집 종료 행사 초기화
    let selectedJoinElements = document.querySelectorAll(".filter-space .end-date-tie .end-date-chk");
    // 각 체크된 요소에 대해 반복하여 체크를 해제합니다.
    selectedJoinElements.forEach((checkbox) => {
        checkbox.checked = false;
    });

    // 필터에 선택되어 보여지는 태그 삭제
    let divToRemove = document.querySelectorAll(".find-filter-bg-three");
    divToRemove.forEach((names) => {
        names.remove();
    });

    // 행사 분야 초기화
    let selectedElementsActivity = document.querySelectorAll(".filter-space .activity-center .activity-chk");
    selectedElementsActivity.forEach((checkbox, i) => {
        checkbox.checked = false;
        categoryCheck[i] = false;
    });

    // 지역 초기화
    let locationTextRadio = document.querySelector(".find-filter-bg-location");
    let locationText = document.getElementById("location-content");
    locationTextRadio.style.display = "none";
    locationText.innerHTML = "";

    const selectElement = document.querySelector(".block-location");
    // 첫 번째 옵션을 기본(default) 값으로 선택합니다.
    selectElement.selectedIndex = 0;

    let page = 1;
    let date = "모든날";
    let region = '';
    let categories = []
    let showFinished = false;
    let ordering = '새 행사순';

    await getList(searchKeyword, page, date, region, categories, showFinished, ordering, showList);
});

//  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// 접은 부분 피기 / 접기 (행사 분야)
let noHide = document.querySelector(".view-btn-tie");
let hide = document.querySelector(".view-btn-tie.hidden");
let hideButton = document.querySelector(".another-view-btn");
const categoryBoxes = document.querySelectorAll(".activity-box");
for (let i=5; i < categoryBoxes.length; i++){
    categoryBoxes[i].classList.add("hidden");
}

hideButton.addEventListener("click", (e) => {
    let closestText = e.target.closest("span");
    if (closestText.getAttribute("value") === "no-hidden") {
        noHide.classList.add("hidden");
        const hiddenCategoryBoxes = document.querySelectorAll(".activity-box.hidden")
        hiddenCategoryBoxes.forEach((box) => {
            box.classList.remove("hidden");
            box.classList.add("showing")
        })
        hide.classList.remove("hidden");
    } else {
        hide.classList.add("hidden");
        const shownCategoryBoxes = document.querySelectorAll(".activity-box.showing")
        shownCategoryBoxes.forEach((box) => {
            box.classList.add("hidden");
            box.classList.remove("showing");
        })
        noHide.classList.remove("hidden");
    }
});

//  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// 슬라이드 배너 (이미 이미지가 존재 하기 때문에 별도로 추가되는 부분은 없음)

const slideContainer = document.querySelector(".swiper-wrap");
const pageCount = document.querySelector(".swiper-page-count");
let count = 0;
// 페이지 생성 시 count 가 초기 상태로 1로 존재 해야하기 때문에 사용
pageCount.innerText = `${count + 1}/2`;

const autoSlide = () => {
    slideContainer.style.transition = "transform 0.4s";
    count++;
    if (count === 2) {
        slideContainer.style.transform = "translate(-" + 100 * (count + 1) + "%)";
        setTimeout(() => {
            slideContainer.style.transition = "transform 0s";
            slideContainer.style.transform = "translate(-100%)";
        }, 500);
        count = 0;
        pageCount.innerText = `${count + 1}/2`;
    } else {
        pageCount.innerText = `${count + 1}/2`;
        slideContainer.style.transform = "translate(-" + 100 * (count + 1) + "%)";
    }
};

slideContainer.style.transform = "translate(-100%)";
let inter = setInterval(autoSlide, 4000);

// 배너 + 선택 시 모달 생성
let plusBtn = document.querySelector(".swiper-more-button");
plusBtn.addEventListener("click", (e) => {
    let thumBanner = document.querySelector(".banner-plus-bg");
    thumBanner.style.display = "block";
});

// 배너 x 선택 시 모달 닫기
let exitBtn = document.querySelector(".absolute-exit-icon");
exitBtn.addEventListener("click", (e) => {
    let thumBanner = document.querySelector(".banner-plus-bg");
    thumBanner.style.display = "none";
});
