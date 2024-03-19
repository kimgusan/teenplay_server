// 공개여부 및 전체를 선택하는 버튼 클릭 시 모달 열기
const tabListBtn = document.querySelector(".tab-list-btn");
const tabList = document.querySelector(".tab-list");
let page = 1
const inner = document.getElementById("inner")
const closeButton = document.querySelector(".close-btn")


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

// 삭제하기 클릭 시 해당 목록 선택하는 이벤트
const replyRemoveBtns = document.querySelectorAll(".reply-remove");
const deleteModalwrap = document.querySelector(".delete-modal-wrap");
let deleteTarget;

replyRemoveBtns.forEach((replyRemoveBtn) => {
    replyRemoveBtn.addEventListener("click", (e) => {

        deleteTarget = replyRemoveBtn;
    });
});

// 모달내 닫기 및 확인 버튼 클릭 시 이벤트
const closeBtns = deleteModalwrap.querySelectorAll("button");



replyService.getList(member_id, 1, status_reply,(reply, total_pages) => {
maxPage = total_pages;
});

const showList = async (reply) =>{
    let text ='';
    if (reply.length ===0){
        text += `<div class="test" style="padding-bottom: 26px">아직 작성한 댓글이 없습니다.</div>`
    }
    console.log(reply)
    reply.forEach((reply)=>{
        if (reply.activity_id){
        text += `<tr class="reply-details">
                    <td class="reply-title">
                        <a href="http://127.0.0.1:10000/activity/detail/?id=${reply.activity_id}">${reply.activity__activity_title}</a>
                    </td>
                    <td class="reply-content">${reply.reply_content}</td>
                    <td class="reply-category">활동 게시글</td>
                    <td class="reply-write-time">${changeDate(reply.created_date)}</td>
                    <td>
                        <button class="reply-remove a${reply.id}" href="">삭제하기</button>
                    </td>
                </tr>`
    }else if (reply.wishlist_id){
            text += `<tr class="reply-details">
                    <td class="reply-title">
                        <a href="/member/mypage-wishlist">${reply.wishlist__wishlist_content}</a>
                    </td>
                    <td class="reply-content">${reply.reply_content}.</td>
                    <td class="reply-category">위시리스트</td>
                    <td class="reply-write-time">${changeDate(reply.created_date)}</td>
                    <td>
                        <button class="reply-remove w${reply.id}" href="">삭제하기</button>
                    </td>
                </tr>`
        }
        else if (reply.club_post_id){
            text += `<tr class="reply-details">
                    <td class="reply-title">
                        <a href="http://127.0.0.1:10000/club/pr-post-detail/?id=${reply.club_post_id}">${reply.club_post__post_title}</a>
                    </td>
                    <td class="reply-content">${reply.reply_content}.</td>
                    <td class="reply-category">홍보 게시글</td>
                    <td class="reply-write-time">${changeDate(reply.created_date)}</td>
                    <td>
                        <button class="reply-remove p${reply.id}" href="">삭제하기</button>
                    </td>
                </tr>`
        }
    }
    )
    return text;
}


const removeButton = document.querySelector(".check-btn")
closeButton.addEventListener("click", ()=>{
    deleteModalwrap.style.display = "none"
})
inner.addEventListener("click", async(e)=>{
    if(e.target.classList[0] === 'reply-remove'){
        deleteModalwrap.querySelector(".check-svg-box").style.display = "none";
        deleteModalwrap.querySelector(".delete-modal-container").style.animation = "popUp 0.5s";
        deleteModalwrap.style.display = "block";
        const updates = e.target.classList[1]
        removeButton.addEventListener("click", async(e)=>{
            replyService.remove(updates)
            page = 1
            currentPage = 1
            const text = await replyService.getList(member_id, currentPage,status_reply, showList);
            inner.innerHTML = text;
            await replyService.getList(member_id, 1, status_reply, async (reply, total_pages) => {
                maxPage = total_pages;
            })
            deleteModalwrap.style.display = "none"
            updatePageButtons();
        })
    }
})





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

    const buttonsToShow = Math.min(maxPage, 5);
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
            const text = await replyService.getList(member_id, currentPage, status_reply, showList);
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

// 서버에서 전체 페이지 수를 가져와서 최대 페이지 업데이트
replyService.getList(member_id, 1, status_reply, async (reply, total_pages) => {
    maxPage = total_pages;
    // 페이지 버튼 다시 업데이트
    updatePageButtons();
});

// 좌 버튼 클릭 이벤트 처리
leftButton.addEventListener('click', async () => {
    if (currentPage > 1) {
        currentPage--;
        const text = await replyService.getList(member_id, currentPage, status_reply, showList);
        inner.innerHTML = text;

        updatePageButtons(); // 페이지 버튼 업데이트
    }
});

// 우 버튼 클릭 이벤트 처리
rightButton.addEventListener('click', async () => {
    if (currentPage < maxPage) {
        currentPage++;
        const text = await replyService.getList(member_id, currentPage, status_reply, showList);
        inner.innerHTML = text;

        updatePageButtons(); // 페이지 버튼 업데이트
    }
});


// 서버에서 전체 페이지 수를 가져와서 최대 페이지 업데이트
replyService.getList(member_id, 1,status_reply,(reply, total_pages) => {
    maxPage = total_pages;
});





replyService.getList(member_id, page,status_reply, showList).then((text) => {
    inner.innerHTML += text;
});




replyService.getList(member_id, page,status_reply, showList);






const changeDate = (dateStr) => {
    const days = ['일', '월', '화', '수', '목', '금', '토'];
    const date = new Date(dateStr); // string타입을 date타입으로 바꿈.
    const month = date.getMonth() + 1; // getMonth()는 0부터 시작하므로 1을 더함.
    const day = date.getDate();
    const dayOfWeek = days[date.getDay()]; // 요일을 숫자로 가져와 해당 요일 문자열로 변환

    // padStart는 ,앞의 숫자의 자릿수 만큼 0을 채워줌
    return `${month.toString().padStart(2, '0')}월 ${day.toString().padStart(2, '0')}일(${dayOfWeek})`;
}
