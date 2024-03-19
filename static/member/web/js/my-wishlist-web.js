let page = 1

// 공개여부 및 전체를 선택하는 버튼 클릭 시 모달 열기
const tabListBtn = document.querySelector(".tab-list-btn");
const tabList = document.querySelector(".tab-list");

document.addEventListener("click", (e) => {
    if (!e.target.closest(".tab-list-contents")) {
        tabList.classList.remove("block");
    } else if (e.target.closest(".tab-list-btn") || e.target.closest(".tab-list")) {
        tabList.classList.toggle("block");
    }
});

// tap을 클릭 시 tabListBtn의 text에 tap을 안의 text로 바꿔준다
const tabs = document.querySelectorAll(".tab");

tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
        tabListBtn.firstElementChild.innerText = tab.innerText;
    });
});

const showList = async (wishlists) => {
    let text = '';
    wishlists.forEach((wishlist) =>  {
        if (wishlists.length ===0) {
            `                                <div class="test">아직 작성한 위시리스트가 없습니다.</div>`
        } else {
            text += `
                                            <tr class="wishlist-details">
                                                <td class="wishlist-content">
                                                    <form action="/wishlist/list/" method="post" name="my-wishlist-form">
                                                        <input type="hidden" name="csrfmiddlewaretoken" id="csrfmiddlewaretoken" value="${csrfToken}">
                                                        <input type="hidden" name="wishlist-id" value="${wishlist.id}" >
                                                        <div class="wishlist-text ${wishlist.id}" data-letter-content="${wishlist.wishlist_content}">${wishlist.wishlist_content}</div>
                                                    </form>
                                                </td>
                                                <td class="wishlist-write-time">${timeForToday(wishlist.created_date)}</td>
                                                <td class="wishlist-disclosure">${wishlist.is_private ? '공개' : '비공개'}</td>
                                                <td class="wishlist-category">${wishlist.category_name}</td>
                                                <td class="wishlist-replayes">${wishlist.reply_total}개</td>
                                                <td class="wishlist-likes">${wishlist.like_total}개</td>
                                                <td>
                                                    <button class="wishlist-remove ${wishlist.id}" href="">삭제하기</button>
                                                </td>
                                            </tr>
        `}

    });
    const myWishlistForms = document.querySelectorAll('form[name=my-wishlist-form]');
    myWishlistForms.forEach((form) => {
        form.addEventListener("click", () => {
            form.submit();
        })
    })
    return text;

};

const leftButton = document.getElementById("leftButton");
const rightButton = document.getElementById("rightButton");
const pageButtonsContainer = document.getElementById("pageButtons");
let currentPage = 1; // 현재 페이지
let maxPage = 1; // 최대 페이지 초기값 설정


// 페이지 버튼을 생성하고 업데이트하는 함수
const updatePageButtons = () => {
    pageButtonsContainer.innerHTML = "";

    const buttonsToShow = Math.min(maxPage, 5);  // 최대 5개의 버튼만 표시
    let startPage = Math.max(1, currentPage - Math.floor(buttonsToShow / 2));
    let endPage = Math.min(maxPage, startPage + buttonsToShow - 1);

    // 마지막 페이지에 도달했을 때, startPage를 조정
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
            const text = await wishlistService.getList(member_id, currentPage,status_wishlist, showList);
            tbody.innerHTML = text;

            pageButton.classList.add("focus-page");

            updatePageButtons(); // 페이지 버튼 업데이트
        });

        pageButtonsContainer.appendChild(pageButton);
    }
};

// 초기 페이지 로드
updatePageButtons();

// 서버에서 전체 페이지 수를 가져와서 최대 페이지 업데이트
wishlistService.getList(member_id, 1, status_wishlist,async (replies, total_pages) => {
    maxPage = total_pages;
    // 페이지 버튼 다시 업데이트
    updatePageButtons();
});

// 좌 버튼 클릭 이벤트 처리
leftButton.addEventListener('click', async () => {
    if (currentPage > 1) {
        currentPage--;
        const text = await wishlistService.getList(member_id, currentPage,status_wishlist, showList);
        tbody.innerHTML = text;

        updatePageButtons(); // 페이지 버튼 업데이트
    }
});

// 우 버튼 클릭 이벤트 처리
rightButton.addEventListener('click', async () => {
    if (currentPage < maxPage) {
        currentPage++;
        const text = await wishlistService.getList(member_id, currentPage,status_wishlist, showList);
        tbody.innerHTML = text;

        updatePageButtons(); // 페이지 버튼 업데이트
    }
});

// 초기 페이지 로드
updatePageButtons();

// 서버에서 전체 페이지 수를 가져와서 최대 페이지 업데이트
wishlistService.getList(member_id, 1, status_wishlist,(wishlists, total_pages) => {
    maxPage = total_pages;
});

const deleteModalwrap = document.querySelector(".delete-modal-wrap");
const deleteBut = document.getElementById('delete-but')
const deleteButclosed = document.getElementById('delete-out')
// const letterModalwrap = document.querySelector(".letter-modal-wrap");
const tbody = document.querySelector(".add-more");

tbody.addEventListener("click", async (e)=>{
    if(e.target.classList[0]==='wishlist-remove'){
        const wishlist_id = e.target.classList[1]
        deleteModalwrap.style.display = 'block'
        deleteBut.addEventListener("click",async (e)=>{
            await wishlistService.remove(wishlist_id);
            page = 1
            const text = await wishlistService.getList(member_id, currentPage,status_wishlist, showList);
            tbody.innerHTML = text;
            deleteModalwrap.style.display = 'none'
        })
    }
});
deleteButclosed.addEventListener("click", async (e) =>{
    deleteModalwrap.style.display = 'none'
})


wishlistService.getList(member_id, page, status_wishlist, showList).then((text) => {
    tbody.innerHTML += text;
});




wishlistService.getList(member_id, page,status_wishlist, showList);

//작성 날짜 계산하기
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
