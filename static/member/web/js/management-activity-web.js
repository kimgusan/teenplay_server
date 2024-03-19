// 공개여부 및 전체를 선택하는 버튼 클릭 시 모달 열기
const sortingBtn = document.querySelector(".sorting-btn");
const sortingListUl = document.querySelector(".sorting-list-ul");
page = 1
const inner = document.querySelector(".activity-activities-contents")

document.addEventListener("click", (e) => {
    if (!e.target.closest(".sorting-box")) {
        sortingListUl.classList.remove("block");
    } else if (e.target.closest(".sorting-btn") || e.target.closest(".sorting-list-ul")) {
        sortingListUl.classList.toggle("block");
    }
});

// tap을 클릭 시 tabListBtn의 text에 tap을 안의 text로 바꿔준다
const sortingTargets = document.querySelectorAll(".sorting-target");

sortingTargets.forEach((sortingTarget) => {
    sortingTarget.addEventListener("click", () => {
        sortingBtn.firstElementChild.innerText = sortingTarget.innerText;
    });
});

const showList = async (activity_data) =>{
    let text = '';
    console.log(activity_data)
    if (activity_data.length ===0){
        text += `<div class="none-actinities-box">
                    <div>
                        <div>작성된 활동이 없습니다.</div>
                    </div>
                </div>`
    }else {activity_data.forEach((activity_data)=>{
        text += `                                            
            <div>
                <div class="top-line"></div>
                <div class="activity-activity-content">
                    <!-- 활동명 (활동 페이지로 이동 링크)-->
                    <div class="activity-activity-name">
                        <a class="activity-activity-link" href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}">${activity_data.activity_title}</a>
                    </div>
                    <!-- 간단소개 -->
                    <div class="activity-activity-hits">${activity_data.activity_intro}</div>
                    <!-- 분야/유형 카테고리 -->
                    <div class="activity-activity-categorys">${activity_data.category__category_name}</div>
                    <!-- 온라인/ 오프라인 경우 지역(구)표시 -->
                    <div class="activity-activity-place"></div>
                    <div class="activity-activity-open-date">${changeDate(activity_data.activity_start)} ~ ${changeDate(activity_data.activity_end)}</div>
                </div>
            </div>`
    })
    }return text;

}

const leftButton = document.getElementById("leftButton");
const rightButton = document.getElementById("rightButton");
const pageButtonsContainer = document.getElementById("pageButtons");
let currentPage = 1;
let maxPage = 1;

const updatePageButtons = () => {
    pageButtonsContainer.innerHTML = "";

    if (maxPage === 0) {
        // 페이지가 없을 경우 버튼 숨김
        leftButton.style.display = "none";
        rightButton.style.display = "none";
        return;
    }

    const buttonsToShow = Math.min(maxPage, 6);
    let startPage = Math.max(1, currentPage - Math.floor(buttonsToShow / 2));
    let endPage = Math.min(maxPage, startPage + buttonsToShow - 1);

    if (endPage === maxPage) {
        startPage = Math.max(1, maxPage - buttonsToShow + 1);
    }

    for (let i = startPage; i <= endPage; i++) {
        const pageButton = document.createElement("button");
        pageButton.classList.add("page-number-btn");

        if (i === currentPage) {
            pageButton.classList.add("focus-page");
        }

        pageButton.setAttribute("type", "button");

        const pageNumberDiv = document.createElement("div");
        pageNumberDiv.classList.add("page-number");
        pageNumberDiv.textContent = i;

        pageButton.appendChild(pageNumberDiv);

        pageButton.addEventListener("click", async () => {
            const previousPageButton = document.querySelector(".focus-page");
            if (previousPageButton) {
                previousPageButton.classList.remove("focus-page");
            }

            currentPage = i;
            const text = await activityService.getList(member_id, currentPage,status_list, showList);
            inner.innerHTML = text;

            pageButton.classList.add("focus-page");

            updatePageButtons(); // 페이지 버튼 업데이트
        });

        pageButtonsContainer.appendChild(pageButton);
    }

    leftButton.style.display = "block"; // 페이지가 있을 경우 좌측 버튼 표시
    rightButton.style.display = "block"; // 페이지가 있을 경우 우측 버튼 표시
};

// 초기 페이지 로드
updatePageButtons();

activityService.getList(member_id, 1, status_list,async (activity_data, total_pages) => {
    maxPage = total_pages;
    // 페이지 버튼 다시 업데이트
    updatePageButtons();
});

// 좌 버튼 클릭 이벤트 처리
leftButton.addEventListener('click', async () => {
    if (currentPage > 1) {
        currentPage--;
        const text = await activityService.getList(member_id, currentPage,status_list, showList);
        inner.innerHTML = text;

        updatePageButtons(); // 페이지 버튼 업데이트
    }
});

// 우 버튼 클릭 이벤트 처리
rightButton.addEventListener('click', async () => {
    if (currentPage < maxPage) {
        currentPage++;
        const text = await activityService.getList(member_id, currentPage,status_list, showList);
        inner.innerHTML = text;

        updatePageButtons(); // 페이지 버튼 업데이트
    }
});


// 서버에서 전체 페이지 수를 가져와서 최대 페이지 업데이트
activityService.getList(member_id, 1,status_list,(activity_data, total_pages) => {
    maxPage = total_pages;
});




activityService.getList(member_id, page,status_list, showList).then((text) => {
    inner.innerHTML += text;
});




activityService.getList(member_id, page,status_list, showList);

const changeDate = (dateStr) => {
    const days = ['일', '월', '화', '수', '목', '금', '토'];
    const date = new Date(dateStr); // string타입을 date타입으로 바꿈.
    const month = date.getMonth() + 1; // getMonth()는 0부터 시작하므로 1을 더함.
    const day = date.getDate();
    const dayOfWeek = days[date.getDay()]; // 요일을 숫자로 가져와 해당 요일 문자열로 변환

    // padStart는 ,앞의 숫자의 자릿수 만큼 0을 채워줌
    return `${month.toString().padStart(2, '0')}월 ${day.toString().padStart(2, '0')}일(${dayOfWeek})`;
}

