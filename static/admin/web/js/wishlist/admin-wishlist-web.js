let page = 1
let category = ""
let keyword = ""
let type = ""
let wishlist_id = ""
let allCheck = false
let totalCountNumber = 0

const CreateService = (() => {
    const showList = (pagination) => {
        console.log(pagination)
        let text = ``;
        pagination.wishlist.wishlist.forEach((page) => {
            text += `
                <li class="main-user-list" data-id="${page.id}">
                    <div class="main-user-list-check">
                        <input type="checkbox" class="main-comment-list-checkbox" id="checkbox" data-id="${page.id}" />
                    </div>
                    <div class="main-user-list-name">${page.member__member_nickname}</div>
                    <div id="title${page.id}" class="main-user-list-status">${page.wishlist_content}</div>
                    <div class="main-user-list-date">${page.wishlist_reply_count}</div>
                    <div class="main-user-list-pay">${page.wishlist_like_count}</div>

                `;
                if(page.is_private === false) {
                    text += `
                            <div class="main-user-list-opne">비공개</div>
                    `
                } else if (page.is_private === true) {
                    text += `
                            <div class="main-user-list-opne">공개</div>
                    `
                }
                text += `
                    <div class="main-user-list-detail">
                        <button class="member-user-list-detail-button toggle-button" data-id="${page.id}">상세보기</button>
                    </div>
                    <input type="hidden" id="category${page.id}" value="${page.category__category_name}">
                </li>
            `;
        })
        return text;
    }

    const showPaging = (pagination) => {
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

        if (pagination.end_page < pagination.real_end) {
            if (pagination.order === 'popular') {
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

        return text;
    }

    // 위시리스트 개수 표기 텍스트
    const CountText = (pagination) => {
        let text = ``;
        text += pagination.total
        totalCountNumber = text;

        return text;
    }

    return {showList: showList, showPaging: showPaging, CountText: CountText}
})();





// ---------------------------------------------------------------------------------------------------------------------
// 위시리스트 게시글 태그
const wishlistData = document.querySelector(".wishlist-data")
// 공지사항 목록 보여주기
function allShowList() {
    adminWishlistService.getPagination(page, category, type, keyword, wishlist_id, CreateService.showList).then((text) => {
        wishlistData.innerHTML = text;
    })
}
allShowList();

// 번호 태그
const mainUserBottomUl = document.querySelector(".main-user-bottom-ul")

// 페이지 번호 보여주기
function allShowPaging() {
    adminWishlistService.getPagination(page, category, type, keyword, wishlist_id, CreateService.showPaging).then((text) => {
        mainUserBottomUl.innerHTML = text;
    })
}
allShowPaging();

// 개수 표기 태그
const totalCount = document.getElementById("total-count");

// 공지사항 개수 표기
const CountShowText = async () => {
    await adminWishlistService.getPagination(page, category, type, keyword, wishlist_id, CreateService.CountText).then((text) => {
        totalCount.textContent = text;
    })
}
CountShowText();





// ---------------------------------------------------------------------------------------------------------------------
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
        allShowList();
    // 페이지 이동 다음 버튼 클릭 시
    } else if (e.target.closest(".main-user-bottom-right")) {
        // 페이지 번호 끝(문자열로 가져오기 때문에 정수로 형변환)에서 + 1 (다음 페이지) 해주기
        page = parseInt(endPage[4].innerText) + 1

        // 페이지에 따른 목록 보여주기
        allShowList();

        // 페이지에 따른 페이지 번호 목록 보여주기
        allShowPaging();
    // 페이지 이동 이전 버튼 클릭 시
    } else if (e.target.closest(".main-user-bottom-left")) {
        // 페이지 번호 끝(문자열로 가져오기 때문에 정수로 형변환)에서 - 1 (이전 페이지) 해주기
        page = parseInt(endPage[0].innerText) - 1

        // 페이지에 따른 목록 보여주기
        allShowList();

        // 페이지에 따른 페이지 번호 목록 보여주기
        allShowPaging();
    }
})





// ---------------------------------------------------------------------------------------------------------------------
// 체크박스
const modalDeleteOpenButtons = document.querySelectorAll(".member-user-list-button");
// 전체 선택 버튼
const statusName = document.querySelector(".main-user-status-name");
// 전체 텍스트
const statusNameText = document.querySelector(".main-user-total-text")

wishlistData.addEventListener('click', (e) => {
    // wishlistBox 요소 중 가까운 조상 중에서 main-user-list 요소 찾기
    // main-user-list가 있으면 옵셔널 체이닝(?.)을 사용하여 프로퍼티에 접근해 main-comment-list-checkbox를 찾기
    const checkboxes = e.target.closest(".main-user-list")?.querySelectorAll(".main-comment-list-checkbox");

    checkboxes.forEach((checkbox) => {
        // console.log(checkbox)
        checkbox.addEventListener('change', () => {
            const checkedItems = document.querySelectorAll(".main-comment-list-checkbox:checked");

            let checkedCount = 0;

            modalDeleteOpenButtons.forEach((deleteButton) => {
                if (checkedItems.length > 0) {
                    deleteButton.classList.remove("disabled");
                    checkedCount = checkedItems.length
                } else if (checkedItems.length === 0) {
                    deleteButton.classList.add("disabled");
                }
            })
            statusNameText.textContent = '전체 중';
            totalCount.textContent = checkedCount;
        });
    })
})

const deleteButton = document.querySelector(".member-user-list-button")

statusName.addEventListener('click', () => {
    let checkboxes = document.querySelectorAll(".main-comment-list-checkbox");
    // 선택된 모든 체크박스 개수를 구합니다.
    const checkedCount = checkboxes.length;

    if (allCheck) {
        allCheck = false;
        statusNameText.textContent = '전체';
        totalCount.textContent = totalCountNumber
    } else {
        allCheck = true;
        statusNameText.textContent = '전체 중';
        // 선택된 모든 체크박스 개수를 텍스트로 표시합니다.
        totalCount.textContent = checkedCount;
    }

    checkboxes.forEach((checkbox) => {
        checkbox.checked = allCheck
    });

    checkboxes = document.querySelectorAll(".main-comment-list-checkbox:checked");

    if (checkboxes.length > 0) {
        // 이부분이 안됩니다
        modalDeleteOpenButtons.forEach((deleteButton) => {
            console.log("if 들어옴")
            deleteButton.classList.remove("disabled");

        });
    } else{
        // 모달 삭제 버튼 활성화 여부를 업데이트합니다.
        modalDeleteOpenButtons.forEach((deleteButton) => {
            console.log("else 들어옴")
            deleteButton.classList.add("disabled");
        });
    }


});




// ---------------------------------------------------------------------------------------------------------------------
// 모달 속 취소 버튼
const modalDeleteCloseButtons = document.querySelectorAll(".admin-user-modal-left-button");
// 모달 속 삭제 버튼
const modalDeleteButtons = document.querySelectorAll(".admin-user-modal-right-button");

// 상태변경
const deletemodal = document.getElementById("admin-user-modal");
const deletemodalBack = document.getElementById("admin-user-modal-backdrop");

let currentTargetLi;

// 삭제하기 버튼 클릭 시 이벤트 발생
modalDeleteOpenButtons.forEach((button) => {
    button.addEventListener("click", (event) => {
        const checkedItems = document.querySelectorAll(".main-comment-list-checkbox:checked");

        // 타겟의 아이디 값 가져오기
        const targetId = event.currentTarget.getAttribute("data-id");
        currentTargetLi = document.querySelector(`li[data-number="${targetId}"]`
        );

        // 모달 열기
        if (checkedItems.length > 0) {
            deletemodal.classList.remove("hidden");
            deletemodalBack.classList.remove("hidden");
        }
    });
});

// 삭제 모달 속 닫기 버튼 클릭 시 이벤트 발생
modalDeleteCloseButtons.forEach((button) => {
    button.addEventListener("click", () => {
        // 삭제 모달 비활성화
        deletemodal.classList.add("hidden");
        deletemodalBack.classList.add("hidden");
    });
});

// 모달 외부를 클릭했을 때 이벤트 처리
document.addEventListener("click", (e) => {
    modalDeleteOpenButtons.forEach((button) => {
        if (!button.contains(e.target) && !deletemodal.contains(e.target)) {
            // 클릭된 요소가 검색 버튼이 아니고 모달 창에 속하지 않으면 모달을 닫음
            deletemodal.classList.add("hidden");
            deletemodalBack.classList.add("hidden");
        }
    });
});

// 삭제 모달 속 삭제 버튼 클릭 시 이벤트 발생
modalDeleteButtons.forEach((button) => {
    //
    button.addEventListener("click", async () => {
        // 체크된 체크 박스 가져오기
        const checkedItems = document.querySelectorAll(".main-comment-list-checkbox:checked");

        // 체크된 체크 박스 반복하여 하나씩 checkbox에 담기
        for (const checkbox of checkedItems) {
            // 체크된 checkbox와 가장 가까운 li 요소를 찾고 data-id 값을 가져오기
            const targetId = checkbox.closest("li").getAttribute("data-id");
            // data-id 속성 값이 같은 li 요소를 가져오기
            await adminWishlistService.remove({ targetId: targetId });
        }

        // 모달 닫기
        deletemodal.classList.add("hidden");
        deletemodalBack.classList.add("hidden");
        allShowList();
        allShowPaging();
        CountShowText();
    });
});





// ---------------------------------------------------------------------------------------------------------------------
// 카테고리
// 카테고리 버튼
const searchOpen = document.querySelector(".main-wish-sellect-button");
// 카테고리 버튼 속 텍스트
const searchText = document.querySelector(".main-wish-sellect-button-span");
// 카테고리 선택 모달
const searchModal = document.querySelector(".admin-message-modal-search");
// 카테고리 모달 속 카테고리 버튼
const searchReceive = document.querySelector(".admin-message-modal-search-receive");
// 카테고리 모달 속 공지사항 버튼
const searchSend = document.querySelector(".admin-message-modal-search-send");
// 카테고리 자주묻는질문 버튼
const searchadd = document.querySelector(".admin-message-modal-search-donotreceive");
// 버튼 이미지
const path = document.querySelector(".main-comment-info-button-svg");

// 검색 버튼 클릭 시 모달 열기
searchOpen.addEventListener("click", () => {
    // 이벤트 전파를 막기 위해 stopPropagation() 호출
    // event.stopPropagation();
    path.setAttribute("transform", "rotate(180)");
    searchModal.classList.remove("hidden");
});

// 모달 외부를 클릭했을 때 이벤트 처리
document.addEventListener("click", (e) => {
    if (!searchOpen.contains(e.target) && !searchModal.contains(e.target)) {
        // 클릭된 요소가 검색 버튼이 아니고 모달 창에 속하지 않으면 모달을 닫음
        path.removeAttribute("transform");
        searchModal.classList.add("hidden");
    }
});

// "전체" 버튼 클릭 시 모달 닫고 텍스트 변경
searchReceive.addEventListener("click", () => {
    path.removeAttribute("transform");
    searchModal.classList.add("hidden");
    searchText.textContent = "전체";
});

// " 활동중" 버튼 클릭 시 모달 닫고 텍스트 변경
searchSend.addEventListener("click", () => {
    path.removeAttribute("transform");
    searchModal.classList.add("hidden");
    searchText.textContent = "공개";
});

// "정지" 버튼 클릭 시 모달 닫고 텍스트 변경
searchadd.addEventListener("click", () => {
    path.removeAttribute("transform");
    searchModal.classList.add("hidden");
    searchText.textContent = "비공개";
});

// 카테고리 버튼 가져오기
const categoryButtons = document.querySelectorAll('.category');
function noticeShowCategory() {
    categoryButtons.forEach((button) => {
        button.addEventListener("click", () => {
            category = button.value;
            page = 1;
            allShowList();
            allShowPaging();
            CountShowText();

            searchInput.value ="";
            keyword = "";
        })
    })
}
noticeShowCategory();





// ---------------------------------------------------------------------------------------------------------------------
// 검색
// 검색 타입(모달 열기 버튼)
const searchType = document.querySelector(".main-message-info-button-add")
// 검색 타입 이름
const seartchTypeText = document.querySelector(".main-message-info-button-text-add")

// 검색 타입 모달
const searchTypeModal = document.querySelector(".admin-message-modal-search-add")
// 검색 타입 모달 속 작성자 버튼
const searchTypePButton = document.querySelector(".admin-message-modal-search-send-add")
// 검색 타입 모달 속 위시리스트 버튼
const searchTypeWButton = document.querySelector(".admin-message-modal-search-receive-add")

// 입력창
const searchInput = document.querySelector(".main-user-info-input")

// 버튼 클릭 시 모달 활성화
searchType.addEventListener('click', () => {
    searchTypeModal.classList.toggle("hidden")
})

// 모달 외부를 클릭했을 때 이벤트 처리
document.addEventListener("click", (e) => {
    if (!searchType.contains(e.target) && !searchTypeModal.contains(e.target)) {
        searchTypeModal.classList.add("hidden");
    }
});

// "작성자" 버튼 클릭 시 모달 닫고 텍스트 변경
searchTypePButton.addEventListener("click", (button) => {
    searchTypeModal.classList.add("hidden");
    seartchTypeText.textContent = "작성자";
});

// " 활동중" 버튼 클릭 시 모달 닫고 텍스트 변경
searchTypeWButton.addEventListener("click", (button) => {
    searchTypeModal.classList.add("hidden");
    seartchTypeText.textContent = "위시리스트";
});


searchInput.addEventListener('keyup', (e) => {
    if (e.keyCode === 13) {
        const typeValue = document.querySelector(".main-message-info-button-text-add")
        if (typeValue.innerHTML === '작성자') {
            type = 'p'
        } else if (typeValue.innerHTML === '위시리스트') {
            type = 'w'
        }

        keyword = e.target.value
        page = 1;
        allShowList();
        allShowPaging();
        CountShowText();
    }
});


// ---------------------------------------------------------------------------------------------------------------------
// 상세 보기
// 상세 추가 태그
const detailModel = document.querySelector(".admin-post-modal");
const detailModelBack = document.querySelector(".detail-box-backdrop");
const detailModelContent = document.querySelector("input[name=title]");
const detailModelCategory = document.querySelector("input[name=category]");
const tag = document.querySelector(".create-tag-list");
const detailBoxClosed = document.querySelector(".admin-user-modal-left-detail-button")

const modifyWishlistTag = (pagination) => {
    console.log('호출됨', pagination)
    const tagDiv = document.createElement("div");
    tagDiv.classList.add("tag-list-add");
    tagDiv.innerHTML = `
        <span style="display: flex; align-items: center;">
            ${pagination}
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14px">
              <line x1="15" y1="5" x2="5" y2="15"></line>
              <line x1="5" y1="5" x2="15" y2="15"></line>
            </svg>
        </span>`;
    tag.appendChild(tagDiv);
}

wishlistData.addEventListener('click', async (e) => {
    if (e.target.classList[0] === 'member-user-list-detail-button') {
        let targetID = e.target.getAttribute("data-id");
        // const wishlistTageData = document.getElementById(`tag${targetID}`).value

        detailModelContent.value = document.getElementById(`title${targetID}`).innerText
        detailModelCategory.value = document.getElementById(`category${targetID}`).value
        tag.innerHTML = ``
        await adminWishlistService.getPagination(page, category, type, keyword, wishlist_id).then(async (data) => {
            const existingTags = await data.wishlist.tags[targetID]
            existingTags.forEach((tag) => {modifyWishlistTag(tag);});
        })

        detailModel.classList.remove("hidden");
        detailModelBack.classList.remove("hidden");
    }
})

detailBoxClosed.addEventListener('click', () => {
    detailModel.classList.add("hidden")
    detailModelBack.classList.add("hidden")
})
