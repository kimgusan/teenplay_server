let page = 1;

const wishlistBox = document.querySelector(".wishlist-data")
const mainUserBottomUl = document.querySelector(".main-user-bottom-ul")

const wishlistCreateService = (() => {
    const showList = (pagination) => {
        let text = ``;
        pagination.pagination.forEach((page) => {
            text += `
                <li class="main-user-list" data-id="${page.wishlist_id}">
                    <div class="main-user-list-check">
                        <input type="checkbox" class="main-comment-list-checkbox" />
                    </div>
                    <div class="main-user-list-name">${page.member_name}</div>
                    <div class="main-user-list-status">${page.wishlist_content}</div>
                    <div class="main-user-list-date">${page.wishlist_like_count}</div>
                    <div class="main-user-list-pay">${page.wishlist_reply_count}</div>
            `;
            if (page.is_private === 0) {
                text += `
                    <div className="main-user-list-opne">비공개</div>
                `
            } else if (page.is_private === 1) {
                text += `
                    <div className="main-user-list-opne">공개</div>
                `
            }
            text += `
                <div class="main-user-list-detail">
                        <button class="member-user-list-detail-button toggle-button" data-target="${page.wishlist_id}">상세보기</button>
                    </div>
                </li>
            `
        })
        return text;
    }

    const showPaing = (pagination) => {
        let text = ``;
        // 시작 페이지가 1보다 큰 경우
        if (pagination.start_page > 1) {
            // 정렬이 popular이라면
            if (pagination.order === 'popular'){
                // 추가
                text += `
                    <li>
                        <a href="${pagination.start_page -1} popular" class="reft main-user-bottom-left">
                            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" class="main-user-bottom-svg">
                                <path fill-rule="evenodd" clip-rule="evenodd" d="M15.58 3.27c.504.405.563 1.115.13 1.587L9.168 12l6.543 7.143a1.076 1.076 0 0 1-.13 1.586 1.26 1.26 0 0 1-1.695-.122L6 12l7.885-8.607a1.26 1.26 0 0 1 1.695-.122Z"></path>
                            </svg>
                        </a>
                    </li>
                 `
            } else {
                text += `
                    <li>
                        <a href="${pagination.start_page -1}" class="reft main-user-bottom-left">
                            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" class="main-user-bottom-svg">
                                <path fill-rule="evenodd" clip-rule="evenodd" d="M15.58 3.27c.504.405.563 1.115.13 1.587L9.168 12l6.543 7.143a1.076 1.076 0 0 1-.13 1.586 1.26 1.26 0 0 1-1.695-.122L6 12l7.885-8.607a1.26 1.26 0 0 1 1.695-.122Z"></path>
                            </svg>
                        </a>
                    </li>
                 `
            }
        }

        // i가 0에서 시작; page_count 보다 작을 때까지 반복; i를 1씩 증가;
        for (let i = 0; i < pagination.page_count; i++) {
            // 현재 반복 횟수 + 시작 페이지 <= 진짜 끝나는 페이지 이하라면
            if (i + pagination.start_page <= pagination.real_end) {
                // 선택된 페이지
                // 페이지가 현재 반복 횟수 + 시작 페이지와 같다면
                if (page === i + pagination.start_page) {
                    // 추가
                    text += `
                        <li class="main-margin">
                            <a href="javascript:void(0)" class="pagination main-user-bottom add-color">
                                <span class="main-user-number add-text-color">${i + pagination.start_page}</span>
                            </a>
                        </li>
                    `
                // 같지 않다면
                } else {
                    // 정렬이 popular 와 같다면
                    if (pagination.order === 'popular') {
                        // 추가
                        text += `
                            <li class="main-margin">
                                <a href="${i + pagination.start_page} popular" class="pagination main-user-bottom" aria-label="page number button">
                                    <span class="main-user-number">${i + pagination.start_page}</span>
                                </a>
                            </li>
                        `
                    // 아니라면
                    } else {
                        // 추가
                        text += `
                            <li class="main-margin">
                                <a href="${i + pagination.start_page}" class="pagination main-user-bottom" aria-label="page number button">
                                    <span class="main-user-number">${i + pagination.start_page}</span>
                                </a>
                            </li>
                        `
                    }
                }
            }
        }

        console.log('들어옴')
        if (pagination.end_page < pagination.real_end) {
            if (pagination.order === 'popular') {
                console.log('들어옴')
                text += `
                    <li class="main-margin">
                        <a href="${pagination.end_page + 1} popular" class="right main-user-bottom-right">
                            <svg class="main-user-bottom-svg" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path fill-rule="evenodd" clip-rule="evenodd" d="M8.42 20.73a1.076 1.076 0 0 1-.13-1.587L14.832 12 8.289 4.857a1.076 1.076 0 0 1 .13-1.586 1.26 1.26 0 0 1 1.696.122L18 12l-7.885 8.607a1.26 1.26 0 0 1-1.695.122Z"></path>
                            </svg>
                        </a>
                    </li>
                `
            } else {
                text += `
                    <li class="main-margin">
                        <a href="${pagination.end_page + 1}" class="right main-user-bottom-right">
                            <svg class="main-user-bottom-svg" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path fill-rule="evenodd" clip-rule="evenodd" d="M8.42 20.73a1.076 1.076 0 0 1-.13-1.587L14.832 12 8.289 4.857a1.076 1.076 0 0 1 .13-1.586 1.26 1.26 0 0 1 1.696.122L18 12l-7.885 8.607a1.26 1.26 0 0 1-1.695.122Z"></path>
                            </svg>
                        </a>
                    </li>
                `
            }
        }
        console.log('들어옴')

        return text;
    }

    // 위시리스트 개수 표기 텍스트
    const wishlistCountText = (pagination) => {
        let text = ``;
        text += pagination.total

        return text;
    }

    return {showList: showList, showPaing: showPaing, wishlistCountText: wishlistCountText}
})()


// 공지사항 목록 보여주기
function wishlistShowList() {
    adminWishlistService.getPagination(page, wishlistCreateService.showList).then((text) => {
        wishlistBox.innerHTML = text;
    })
}
wishlistShowList();

// 페이지 번호 보여주기
function wishlistShowPaging() {
    adminWishlistService.getPagination(page, wishlistCreateService.showPaging).then((text) => {
        mainUserBottomUl.innerHTML = text;
    })
}
wishlistShowPaging();

// 페이지 네이션
// 페이지 번호 박스 클릭 시 이벤트 발생
mainUserBottomUl.addEventListener("click", (e) => {
    // 페이지 번호 a태그
    const mainUserBottom = document.querySelectorAll(".main-user-bottom")
    // 페이지 번호 박스 속 번호
    const endPage = document.querySelectorAll(".main-user-number")

    // 페이지 이동 막아주기
    e.preventDefault()

    // 만약, 페이지 번호 클릭 시
    if (e.target.closest(".main-user-bottom") && e.target.closest(".main-user-bottom").classList.contains('pagination')) {
        console.log("if 들어옴")

        // 기존 선택된 페이지 번호 스타일 삭제
        mainUserBottom.forEach((userBottom) => {
            userBottom.classList.remove("add-color")
        })
        endPage.forEach((userNumber) => {
            userNumber.classList.remove("add-text-color")
        })

        // 새롭게 선택된 페이지 번호 스타일 부여
        e.target.closest(".main-user-bottom").classList.add("add-color")
        e.target.closest(".main-user-bottom").querySelector(".main-user-number").classList.add("add-text-color")

        // 페이지 번호를 텍스트로 가져와 page로 담기
        page = e.target.closest(".main-user-bottom").querySelector(".main-user-number").innerText

        // 번호에 따른 게시글 목록 가져오기
        wishlistShowList();
    // 페이지 이동 다음 버튼 클릭 시
    } else if (e.target.closest(".main-user-bottom-right")) {
        // 페이지 번호 끝(문자열로 가져오기 때문에 정수로 형변환)에서 + 1 (다음 페이지) 해주기
        page = parseInt(endPage[4].innerText) + 1

        // 페이지에 따른 목록 보여주기
        wishlistShowList();

        // 페이지에 따른 페이지 번호 목록 보여주기
        wishlistShowPaging();
    // 페이지 이동 이전 버튼 클릭 시
    } else if (e.target.closest(".main-user-bottom-left")) {
        // 페이지 번호 끝(문자열로 가져오기 때문에 정수로 형변환)에서 - 1 (이전 페이지) 해주기
        page = parseInt(endPage[0].innerText) - 1

        // 페이지에 따른 목록 보여주기
        wishlistShowList();

        // 페이지에 따른 페이지 번호 목록 보여주기
        wishlistShowPaging();
    }
})

// 공지사항 개수 표기
function noticeCountShowText() {
    adminWishlistService.getPagination(page, createListService.noticeCountText).then((text) => {
        totalCount.textContent = text;
    })
}
noticeCountShowText();