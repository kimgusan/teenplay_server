const clubNoticeDelete = document.querySelector(".club-notice-delete");
const clubNoticeCheckboxes = document.querySelectorAll("#club-notice-checkbox");
    const allChecked = document.querySelector("#all-checked");
const checkedFn = () => {
    // checkbox가 1개 이상 checked일 경우 공지 삭제 버튼 활성화
    const clubNoticeCheckboxes = document.querySelectorAll("#club-notice-checkbox");
    const allChecked = document.querySelector("#all-checked");

    clubNoticeCheckboxes.forEach((clubNoticeCheckbox) => {
        clubNoticeCheckbox.addEventListener("input", () => {
            const clubNoticeCheckedCheckboxes = document.querySelectorAll("#club-notice-checkbox:checked");

            // 체크된 체크박스가 없다면 공지 삭제 비활성화, 있다면 활성화
            disabledBtn();

            // 체크된 체크박스 개수가 전체 체크박스 개수와 같다면 allChecked 체크상태로, 아니라면 풀린상태로
            if (clubNoticeCheckedCheckboxes.length === clubNoticeCheckboxes.length) {
                allChecked.checked = true;
            } else {
                allChecked.checked = false;
            }
        });
    });
}

const allCheckedFn = () => {
    const clubNoticeCheckboxes = document.querySelectorAll("#club-notice-checkbox");
    const allChecked = document.querySelector("#all-checked");
    // 항목의 체크박스 클릭 시 모든 체크박스에 발생하는 이벤트
    allChecked.addEventListener("input", (e) => {
        clubNoticeCheckboxes.forEach((clubNoticeCheckbox) => {
            clubNoticeCheckbox.checked = e.target.checked;
        });

        disabledBtn();
    });
}


// 공지 삭제 클릭 시 모달을 표시는 이벤트
const deleteModalWrap = document.querySelector(".delete-modal-wrap");
const deleteModalContainer = deleteModalWrap.querySelector(".delete-modal-container");

clubNoticeDelete.addEventListener("click", () => {
    deleteModalContainer.style.animation = "popUp 0.5s";
    deleteModalWrap.style.display = "block";
});

// 삭제 모달 내 버튼 클릭 시 발생하는 이벤트
const deleteCheckModalContainer = deleteModalWrap.querySelector(".delete-check-modal-container");
const deleteModalBtns = deleteModalWrap.querySelectorAll("button");

deleteModalBtns.forEach((deleteModalBtn) => {
    deleteModalBtn.addEventListener("click", (e) => {
        deleteModalContainer.style.animation = "popDown 0.5s";
        if (e.target.className === "delete-modal-cancle-btn") {
            setTimeout(() => {
                deleteModalWrap.style.display = "none";
            }, 450);
        } else if (e.target.className === "delete-btn") {
            let clubNoticeCheckedCheckboxes = document.querySelectorAll("#club-notice-checkbox:checked");

            clubNoticeCheckedCheckboxes.forEach((clubNoticeCheckedCheckbox) => {
                clubNoticeCheckedCheckbox.closest(".club-notice-info").remove();
            });

            allChecked.checked = false;
            disabledBtn();

            setTimeout(() => {
                deleteModalContainer.style.display = "none";
                deleteCheckModalContainer.style.animation = "popUp 0.5s";
                deleteCheckModalContainer.style.display = "flex";
            }, 450);
        } else {
            deleteCheckModalContainer.style.animation = "popDown 0.5s";
            setTimeout(() => {
                deleteCheckModalContainer.style.removeProperty("animation");
                deleteCheckModalContainer.style.display = "none";
                deleteModalContainer.style.display = "flex";
                deleteModalWrap.style.display = "none";
            }, 450);
        }
    });
});

// 공지 삭제 비활성화 함수
const disabledBtn = () => {
    const clubNoticeCheckedCheckboxes = document.querySelectorAll("#club-notice-checkbox:checked");

    // 체크된 체크박스가 없다면 공지 삭제 비활성화, 있다면 활성화
    if (clubNoticeCheckedCheckboxes.length === 0) {
        clubNoticeDelete.disabled = true;
    } else {
        clubNoticeDelete.disabled = false;
    }
};


const noticeDetailViewFn = () => {
    //  제목 클릭 시 상세보기가 나오고 나와 있을 경우 끄는 이벤트
    const clubNoticeTitles = document.querySelectorAll(".club-notice-title");

    clubNoticeTitles.forEach((clubNoticeTitle) => {
        clubNoticeTitle.addEventListener("click", (e) => {
            let targetSvg = clubNoticeTitle.querySelector(".club-notice-show-icon");

            if (targetSvg.classList.contains("true")) {
                targetSvg.classList.remove("true");
                targetSvg.removeAttribute("transform");
                clubNoticeTitle.closest(".club-notice-info").querySelector(".club-notice-content-box").removeAttribute("style");
                return;
            }
            targetSvg.classList.add("true");
            clubNoticeTitle.querySelector(".club-notice-show-icon").setAttribute("transform", "rotate(180)");
            clubNoticeTitle.closest(".club-notice-info").querySelector(".club-notice-content-box").style.display = "block";
        });
    });

}


const clubNoticeList = document.querySelector('.club-notice-list')

// 버튼에 클릭 이벤트 리스너 추가

const listDeleteBtnFn = () => {
    clubNoticeDelete.addEventListener('click', () => {
    let del_item = [];
    const checkboxes = document.querySelectorAll('.club-notice-checkbox');
    const checkedCheckboxes = Array.from(checkboxes).filter(checkbox => checkbox.checked);
    checkedCheckboxes.forEach(checkbox => {
        const clubNoticeInfo = checkbox.closest('.club-notice-info');
        if (clubNoticeInfo) {
            del_item.push(clubNoticeInfo.classList[1])
        }
    });
    const deleteBtn = document.querySelector('.delete-btn');
    deleteBtn.addEventListener('click', async () => {
        await mypageClubNoticeListService.list(club_id, page, del_item, showList);
    })
});
}



const addPaginationEvent = async (pageInfo) => {
    const prevBtn = document.querySelector(".prev-btn")
    const nextBtn = document.querySelector(".next-btn")
    const pageNumberBtn = document.querySelectorAll(".page-number-btn")
    const pageNumber = document.querySelectorAll(".page-number")

    prevBtn.addEventListener("click", async () => {
        if (pageInfo.currentPage <= 1) return;
        await mypageActivityListService.list(club_id, --page, del_item, showList);
    })

    nextBtn.addEventListener("click", async () => {
        if (pageInfo.currentPage === pageInfo.realEnd) return;
        await mypageActivityListService.list(club_id, ++page, del_item, showList);
    })

    pageNumberBtn.forEach((btn, i) => {
        btn.addEventListener("click", async () => {
            page = pageNumber[i].innerHTML;
            await mypageClubNoticeListService.list(club_id, page, del_item, showList);
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

const showList = (orderList) => {
    let text = ``;
    let pageInfo = orderList.pop()
    console.log(pageInfo)
    if (orderList.length === 0) {
        text += ``
    } else {
        for (let item of orderList) {
            text += `
            <div class="club-notice-info ${item.id}">
                <div class="club-notice-item">
                    <div class="club-notice-checkbox-wrap">
                        <div class="club-notice-checkbox-container">
                            <div class="club-notice-checkbox-box">
                                <input id="club-notice-checkbox" class="club-notice-checkbox"
                                       type="checkbox"/>
                            </div>
                        </div>
                    </div>
                    <div class="club-notice-sequence">${item.id}</div>
                    <div class="club-notice-title">
                        <div>${item.notice_title}</div>
                        <svg xmlns="http://www.w3.org/2000/svg" class="club-notice-show-icon"
                             fill="none" viewBox="0 0 24 24" stroke="currentColor"
                             stroke-width="2">
                            <path stroke-linecap="round" stroke-linejoin="round"
                                  d="M19 9l-7 7-7-7"></path>
                        </svg>
                    </div>
                    <div class="club-notice-date">${item.created_date}</div>
                    <div class="club-notice-modify">
                        <!-- 수정하기 링크 필요 -->
                        <a class="club-notice-modify-link"
                           href="/member/mypage-notice-modify/${club_id}/?id=${item.id}">수정</a>
                    </div>
                </div>
                <div class="club-notice-content-box">
                    <div class="club-notice-content">${item.notice_content}</div>
                </div>
            </div>
        `
        }
    }
    clubNoticeList.innerHTML = text;
    showPagination(pageInfo)
    noticeDetailViewFn()
    checkedFn()
    allCheckedFn()
    listDeleteBtnFn()
}


mypageClubNoticeListService.list(club_id, page, del_item, showList);