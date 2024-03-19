if(page === ''){
    page = 1;
}
// let category = '';
// let ordering = '최신순';
// let keyword = '';

const searchTotalCount = document.querySelector(".search-total-count")
const prContentContainer = document.querySelector(".pr-content-container")
const prPagination = document.querySelector(".pr-pagination")

const addPaginationEvent = (pageInfo) => {
    const prPaginationPrev = document.querySelector(".pr-pagination-prev")
    const prPaginationNext = document.querySelector(".pr-pagination-next")
    const prPaginationPages = document.querySelectorAll(".pr-pagination-page")
    const prPaginationPageNumbers = document.querySelectorAll(".pr-pagination-page-number")

    prPaginationPrev.addEventListener("click", async () => {
        if (pageInfo.currentPage <= 1) return;
        await clubPostListService.getList(--page, category, order, keyword, showList);
    })

    prPaginationNext.addEventListener("click", async () => {
        if (pageInfo.currentPage === pageInfo.realEnd) return;
        await clubPostListService.getList(++page, category, order, keyword, showList);
    })

    prPaginationPages.forEach((btn, i) => {
        btn.addEventListener("click", async () => {
            page = prPaginationPageNumbers[i].innerHTML;
            await clubPostListService.getList(page, category, order, keyword, showList);
        })
    })
}

const showPagination = (pageInfo) => {
    const totalCount = pageInfo.totalCount;
    const startPage = pageInfo.startPage;
    const endPage = pageInfo.endPage;
    const currentPage = pageInfo.page;
    const realEnd = pageInfo.realEnd;
    let pageText = ``;

    if (totalCount === 0) {
        prPagination.innerHTML = ``;
    } else{
        prPagination.innerHTML = `
            <!-- 1페이지일 때 이전버튼 비활성화(disabled 클래스 추가) -->
            <button class="pr-pagination-prev ${currentPage === 1 ? 'disabled' : ''}" type="button">
                <svg xmlns="http://www.w3.org/2000/svg" class="pr-pagination-prev-icon" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                </svg>
            </button>
            <div class="pr-pagination-page-wrap">
        `;
        for (let i = startPage; i <= endPage; i++){
            prPagination.innerHTML += `
                <!-- 현재 페이지일 때 active 클래스 추가 -->
                <button class="pr-pagination-page ${currentPage === i ? 'active' : ''}" type="button">
                    <div class="pr-pagination-page-number">${i}</div>
                </button>
            `;
        }
        prPagination.innerHTML += `
            </div>
            <button class="pr-pagination-next ${currentPage === realEnd ? 'disabled' : ''}" type="button">
                <svg xmlns="http://www.w3.org/2000/svg" class="pr-pagination-next-icon" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd"></path>
                </svg>
            </button>
        `;
        addPaginationEvent(pageInfo);
    }
}

const showList = (clubPostList) => {
    let text = ``;
    let pageInfo = clubPostList.pop()
    searchTotalCount.innerText = pageInfo.totalCount

    if (clubPostList.length === 0) {
        text += ``
    } else{
        for (let clubPost of clubPostList) {
            text += `
                <!-- 아래 div가 게시물마다 반복 -->
                <div class="pr-content-boxes">
                    <!-- 모임 홍보글 이미지 -->
                    <a href="/club/pr-post-detail/?id=${clubPost.id}&page=${page}&category=${category}&order=${order}&keyword=${keyword}" class="pr-content-img-wrap" target="_self">
                        <img src="/upload/${clubPost.image_path}" class="pr-content-img" />
                    </a>
                    <div class="pr-content-text-wrap">
                        <div class="pr-content-text-container">
                            <div class="pr-content-text-top-wrap">
                                <!-- 카테고리 -->
                                <div class="pr-content-field">${clubPost.category_name}</div>
                                <!-- 모임 이름 -->
                                <div class="pr-content-text-name">${clubPost.club_name}</div>
                            </div>
                            <!-- 홍보글 제목 -->
                            <div class="pr-content-text-title-wrap">
                                <a href="/club/pr-post-detail/?id=${clubPost.id}&page=${page}&category=${category}&order=${order}&keyword=${keyword}" class="pr-content-text-title-link">
                                    <p class="pr-content-text-title">${clubPost.post_title}</p>
                                </a>
                            </div>
                            <!-- 홍보글 내용 -->
                            <div class="pr-content-text-content-wrap">
                                <p class="pr-content-text-content">${clubPost.post_content}</p>
                            </div>
                            <!-- 구성원, 댓글 개수, 작성시간, 조회수 -->
                            <div class="pr-content-info-wrap">
                                <div class="pr-content-info-left-wrap">
                                    <span class="pr-content-info-left-title"> 구성원 </span>
                                    <span class="pr-content-info-left-count"> ${clubPost.club_member_count} </span>
                                </div>
                                <div class="pr-content-info-right-wrap">
                                    <span class="pr-content-info-right-replies"> 댓글 ${clubPost.reply_count} </span>
                                    <span class="pr-content-info-border">|</span>
                                    <span class="pr-content-info-right-date">${timeForToday(clubPost.created_date)}</span>
                                    <span class="pr-content-info-border">|</span>
                                    <span class="pr-content-info-right-views">조회 ${clubPost.view_count}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `
        }
    }
    prContentContainer.innerHTML = text;
    showPagination(pageInfo)
}

// 카테고리 선택 시 선택한 카테고리에 대한 목록을 보여주는 부분
const prPostCategoryBox = document.querySelector(".pr-post-category-box")

prPostCategoryBox.addEventListener("change", async (e) => {
    category = e.target.value
    prPostSearchInput.value = ''
    keyword = ''
    page = 1
    await clubPostListService.getList(page, category, order, keyword, showList)
})

// 검색 창에 enter 키 누를 경우 입력한 value를 keyword로 검색하는 기능
const prPostSearchInput = document.querySelector(".pr-post-search-input")

prPostSearchInput.addEventListener("keyup", async (e) => {
    if (e.keyCode === 13) {
        page = 1
        keyword = prPostSearchInput.value
        await clubPostListService.getList(page, category, order, keyword, showList)
    }
})

// 정렬순 선택 시 class를 추가 및 삭제 후 선택한 것에 맞게 정렬되어 보여주는 기능
const orderingBtns = document.querySelectorAll(".ordering-wrap button")
const newOrdering = document.querySelector(".new-ordering")
const hitsOrdering = document.querySelector(".hits-ordering")

orderingBtns.forEach((btn) => {
    btn.addEventListener("click", async (e) => {
        if (e.target.closest(".new-ordering")) {
            newOrdering.classList.add("selected-ordering")
            hitsOrdering.classList.remove("selected-ordering")
            order = newOrdering.innerText
        } else if (e.target.closest(".hits-ordering")) {
            newOrdering.classList.remove("selected-ordering")
            hitsOrdering.classList.add("selected-ordering")
            order = hitsOrdering.innerText
        }
        page = 1
        await clubPostListService.getList(page, category, order, keyword, showList)
    })
})

if (keyword) {
    prPostSearchInput.value = keyword
}

const options = document.querySelectorAll("option")

if (category) {
    prPostCategoryBox.value = category
}

if (order === '최신순') {
    newOrdering.classList.add("selected-ordering")
    hitsOrdering.classList.remove("selected-ordering")
} else{
    newOrdering.classList.remove("selected-ordering")
    hitsOrdering.classList.add("selected-ordering")
}

clubPostListService.getList(page, category, order, keyword, showList)




















function timeForToday(datetime) {
    const today = new Date();
    const date = new Date(datetime);

    let gap = Math.floor((today.getTime() - date.getTime()) / 1000 / 60);

    if (gap < 1) {
        return "방금 전";
    }

    if (gap < 60) {
        return `${gap}분 전`;
    }

    gap = Math.floor(gap / 60);

    if (gap < 24) {
        return `${gap}시간 전`;
    }

    gap = Math.floor(gap / 24);

    if (gap < 31) {
        return `${gap}일 전`;
    }

    gap = Math.floor(gap / 31);

    if (gap < 12) {
        return `${gap}개월 전`;
    }

    gap = Math.floor(gap / 12);

    return `${gap}년 전`;
}