// 공개여부 및 전체를 선택하는 버튼 클릭 시 모달 열기
const sortingBtn = document.querySelector(".sorting-btn");
const sortingListUl = document.querySelector(".sorting-list-ul");

document.addEventListener("click", (e) => {
    if (!e.target.closest(".sorting-box")) {
        sortingListUl.classList.remove("block");
    } else if (e.target.closest(".sorting-btn") || e.target.closest(".sorting-list-ul")) {
        sortingListUl.classList.toggle("block");
    }
});

// tap을 클릭 시 tabListBtn의 text에 tap을 안의 text로 바꿔준다


const addPaginationEvent = async (pageInfo) => {
    const prevBtn = document.querySelector(".prev-btn")
    const nextBtn = document.querySelector(".next-btn")
    const pageNumberBtn = document.querySelectorAll(".page-number-btn")
    const pageNumber = document.querySelectorAll(".page-number")

    prevBtn.addEventListener("click", async () => {
        if (pageInfo.currentPage <= 1) return;
        await mypageActivityListService.list(club_id, --page, order, showList);
    })

    nextBtn.addEventListener("click", async () => {
        if (pageInfo.currentPage === pageInfo.realEnd) return;
        await mypageActivityListService.list(club_id, ++page, order, showList);
    })

    pageNumberBtn.forEach((btn, i) => {
        btn.addEventListener("click", async () => {
            page = pageNumber[i].innerHTML;
            await mypageActivityListService.list(club_id, page, order, showList);
        })
    })
}
const paginationBox = document.querySelector('.pagination-box')
const showPagination = (pageInfo) => {
    const totalCount = pageInfo.totalCount;
    const startPage = pageInfo.startPage;
    const endPage = pageInfo.endPage;
    const currentPage = pageInfo.page;
    const realEnd = pageInfo.realEnd;
    let pageText = ``;
    console.log(totalCount, startPage, endPage, currentPage, realEnd)
    if (totalCount === 0) {
        paginationBox.innerHTML = ``;
    } else {
        paginationBox.innerHTML = `
               <button class="prev-btn ${currentPage === 1 ? 'disabled' : ''}" type="button">
                    <svg xmlns="http://www.w3.org/2000/svg" class="left-svg"
                         viewBox="0 0 20 20"
                         fill="currentColor">
                        <path fill-rule="evenodd"
                              d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z"
                              clip-rule="evenodd"></path>
                    </svg>
                </button>
                <div class="pagination-number-box">
        `;
        for (let i = startPage; i <= endPage; i++) {
            paginationBox.innerHTML += `
          <button class="page-number-btn ${currentPage === i ? 'seleced-page-number-btn' : ''}" type="button">
                <div class="page-number">${i}</div>
            </button>
            `;
        }
        paginationBox.innerHTML += `
             </div>
                <button class="next-btn ${currentPage === realEnd ? 'disabled' : ''}" type="button">
                    <svg xmlns="http://www.w3.org/2000/svg" class="right-svg"
                         viewBox="0 0 20 20"
                         fill="currentColor">
                        <path fill-rule="evenodd"
                              d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                              clip-rule="evenodd"></path>
                    </svg>
                </button>
        `;
        addPaginationEvent(pageInfo);
    }
}

const sortingTargets = document.querySelectorAll(".sorting-target");
const clubActivitiesContents = document.querySelector('.club-activities-contents')
const showList = (orderList) => {
    let text = ``;
    let pageInfo = orderList.pop()
    console.log(pageInfo)
    if (orderList.length === 0) {
        text += ``
    } else {
        for (let item of orderList) {
            text += `
            <div>
                <div class="top-line"></div>
                <div class="club-activity-content">
                    <!-- 활동 id -->
                    <div class="club-activity-number">${item.id}</div>
                    <!-- 활동 조회수 -->
                    <div class="club-activity-hits">4</div>
                    <div class="club-activity-title">
                        <!-- 활동 관리 페이지 주소, 제목 필요 -->
                        <a class="club-activity-link" href="/member/activity/?activity_id=${ item.id }">${item.activity_title}</a>
                    </div>
                    <div class="club-activity-date">${item.activity_start}</div>
                    <div class="club-activity-open-date">${item.created_date}</div>
                </div>
            </div>
        `
        }
    }
    clubActivitiesContents.innerHTML = text;
    showPagination(pageInfo)
}

sortingTargets.forEach((sortingTarget) => {
    sortingTarget.addEventListener("click", async () => {
        sortingBtn.firstElementChild.innerText = sortingTarget.innerText;
        order = sortingBtn.firstElementChild.innerText
        page = 1
        await mypageActivityListService.list(club_id, page, order, showList)
    });
});


mypageActivityListService.list(club_id, page, order, showList);