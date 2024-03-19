// 전체 선택 버튼
const statusName = document.querySelector(".main-user-status-name");
// 체크박스
const checkBox = document.querySelectorAll(".main-comment-list-checkbox");
// 선택된 게시글 개수
const totalCount = document.getElementById("total-count");
// 상태 변경 버튼
const deleteButton = document.querySelector(".toggle-button")
// 목록 태그
const noticeBox = document.querySelector(".notice-data")
// 페이지 번호 박스
const pageNumberBox = document.querySelector(".main-user-bottom-ul")
// 페이지 번호
const pageNumberButtons = document.querySelectorAll(".main-user-bottom")
// 페이지 이동 이전 버튼
const prevButton = document.querySelector(".main-user-bottom-reft")
// 페이지 이동 다음 버튼
const nextButton = document.querySelector(".main-user-bottom-right")

let page = 1;
let pageNumber = 1;

// 게시글 목록 가져오기
const showList = (pagination) => {
    let text = ``;
    pagination.pagination.forEach((page) => {
        text += `
            <li class="main-user-list" data-id="${page.id}">
                <div class="main-comment-list-check">
                    <input type="checkbox" class="main-comment-list-checkbox" />
                </div>
                <div class="main-user-list-name">${page.notice_title}</div>
                <div class="main-user-list-status">${page.created_date}</div>
        `;
        if(page.notice_type === 0) {
            text += `
                <div class="main-user-list-category" >공지사항</div>
            `
        } else if (page.notice_type === 1) {
            text += `
                <div class="main-user-list-category" >자주묻는질문</div>
            `
        }
        text += `
            <div class="main-user-list-detail">
                    <button class="member-user-list-detail-button toggle-button" data-target="${page.id}">상세보기</button>
                </div>
            </li>
        `
    })
    return text;
}

const showPageing = (pagination) => {
    let text = ``;
    // 시작 페이지가 1보다 큰 경우
    if (pagination.start_page > 1) {
        // 정렬이 popular이라면
        if (pagination.order === 'popular'){
            // 추가
            text += `
                <li>
                    <a href="${pagination.start_page -1} popular" class="reft main-user-bottom-reft">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" class="main-user-bottom-svg">
                            <path fill-rule="evenodd" clip-rule="evenodd" d="M15.58 3.27c.504.405.563 1.115.13 1.587L9.168 12l6.543 7.143a1.076 1.076 0 0 1-.13 1.586 1.26 1.26 0 0 1-1.695-.122L6 12l7.885-8.607a1.26 1.26 0 0 1 1.695-.122Z"></path>
                        </svg>
                    </a>
                </li>
             `
        } else {
            text += `
                <li>
                    <a href="${pagination.start_page -1}" class="reft main-user-bottom-reft">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" class="main-user-bottom-svg">
                            <path fill-rule="evenodd" clip-rule="evenodd" d="M15.58 3.27c.504.405.563 1.115.13 1.587L9.168 12l6.543 7.143a1.076 1.076 0 0 1-.13 1.586 1.26 1.26 0 0 1-1.695-.122L6 12l7.885-8.607a1.26 1.26 0 0 1 1.695-.122Z"></path>
                        </svg>
                    </a>
                </li>
             `
        }
    }

    // x문자열 선언
    const x = "x"

    // i가 0에서 시작; page_count 보다 작을 때까지 반복; i를 1씩 증가;
    for (let i = 0; i < pagination.page_count; i++) {
        // 반복문을 사용하여 문자열 x를 오른쪽으로 정렬
        // const padX = x.padEnd(pagination.page_count, ' ');
        // 현재 반복 횟수 + 시작 페이지 <= 진짜 끝나는 페이지 이하라면
        if (i + pagination.start_page <= pagination.real_end) {
            // 페이지가 현재 반복 횟수 + 시작 페이지와 같다면
            if (page === i + pagination.start_page) {
                // 추가
                text += `
                    <li class="main-margin">
                        <a href="javascript:void(0)" class="pagination main-user-bottom">
                            <span class="main-user-number" style="color: #CE201B;">${i + pagination.start_page}</span>
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

// 체크 박스 체크 시 카운트 업데이트 함수 선언
function checkBoxCount() {
    // 체크박스 체크 시 전체 숫자 up
    let checkedCount = 0;

    // 체크박스 반복하여 하나씩 가져오기
    checkBox.forEach((check) => {
        // 체크박스 체크 시 체크 카운트 up
        if (check.checked) {
            checkedCount++;
        }
    });

    // 선택된 회원의 개수(텍스트)에 checkedCount(개수 데이터) 업데이트
    totalCount.textContent = checkedCount;
}

// 체크 박스 선택 시 함수 실행
// 체크 박스 반복하여 하나씩 가져오기
checkBox.forEach((check) => {
    // 체크 박스 값 변경(change) 시 이벤트 발생, checkBoxCount 함수 실행
    check.addEventListener('change', checkBoxCount)
});

// 체크 박스 전체 선택 버튼 기능 구현
// 전체 선택 버튼 클릭 시 이벤트 발생
statusName.addEventListener("click", () => {
    // 모든 체크박스가 체크되었다고 가정
    let allChecked = true;

    // 체크 박스 반복하여 하나씩 가져오기
    checkBox.forEach((checkbox) => {
        // 만약, 체크 박스가 체크되어 있지 않다면
        if (!checkbox.checked) {
                allChecked = false;
                // 체크박스 체크해주기
                checkbox.checked = true;
                // 상태 변경 버튼 활성화
                deleteButton.classList.remove("disabled");
            }
        });
    checkBoxCount();

    // 체크 박스 반복하여 하나씩 가져오기
    checkBox.forEach((checkbox) => {
        // 모든 체크 박스가 체크 되어 있다면
        if (allChecked) {
            // 체크박스 체크 없애기
            checkbox.checked = false;
            // 상태 변경 버튼 비활성화
            deleteButton.classList.add("disabled");
        }
    });
    // checkBoxCount 함수 실행
    checkBoxCount();
});

// 체크 박스 선택 여부에 따라 버튼 활성화/비활성화
// 체크 박스 반복하여 하나씩 가져오기
checkBox.forEach((checkbox) => {
    // 체크 박스 값 변경(change) 시 이벤트 발생, checkBoxCount 함수 실행
    checkbox.addEventListener('change', () => {
        // 체크 박스 선택 여부에 따라 버튼 활성화/비활성화
        deleteButton.classList.toggle("disabled", !checkbox.checked);
    });
});

// 삭제하기 버튼
const modalDeleteOpenButtons = document.querySelectorAll(".member-user-list-button");
// 모달 속 취소 버튼
const modalDeleteCloseButtons = document.querySelectorAll(".admin-user-modal-left-button");
// 모달 속 삭제 버튼
const modalDeleteButtons = document.querySelectorAll(".admin-user-modal-right-button");

// 삭제 모달
const deletemodal = document.getElementById("admin-user-modal");
const deletemodalBack = document.getElementById("admin-user-modal-backdrop");

let currentTargetLi;

// 삭제하기 버튼 클릭 시 이벤트 발생
modalDeleteOpenButtons.forEach((button) => {
    button.addEventListener("click", (event) => {
        // 타겟의 아이디 값 가져오기
        const targetId = event.currentTarget.getAttribute("data-target");
        currentTargetLi = document.querySelector(`li[data-number="${targetId}"]`
        );

        // 모달 열기
        deletemodal.classList.remove("hidden");
        deletemodalBack.classList.remove("hidden");
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

// 삭제 모달 속 삭제 버튼 클릭 시 이벤트 발생
modalDeleteButtons.forEach((button) => {
    button.addEventListener("click", () => {
        // 체크된 체크 박스 가져오기
        const checkedItems = document.querySelectorAll(".main-comment-list-checkbox:checked");

        // 체크된 체크 박스 반복하여 하나씩 checkbox에 담기
        for (const checkbox of checkedItems) {
            // 체크된 checkbox와 가장 가까운 li 요소를 찾고 data-id 값을 가져오기
            const targetId = checkbox.closest("li").getAttribute("data-id");

            // data-id 속성 값이 같은 li 요소를 가져오기
            adminNoticeService.remove({ targetId: targetId });
        }

        // 모달 닫기
        deletemodal.classList.add("hidden");
        deletemodalBack.classList.add("hidden");
        adminNoticeService.getList(showList).then((text) => {
            noticeBox.innerHTML = text;
        })
    });
});

// 페이지네이션
// 페이지 버튼 보여주기
adminNoticeService.getPagination(page, pageNumber, showPageing).then((text) => {
    pageNumberBox.innerHTML = text;
})

// 글 목록 가져오기
adminNoticeService.getPagination(page, pageNumber, showList).then((text) => {
    noticeBox.innerHTML = text;
})

// // 각 페이지 데이터 호출하기
// pageNumberButtons.forEach((pageNumberButton) => {
//     pageNumberButton.addEventListener("click", (e) => {
//         adminNoticeService.getPagination(page, pageNumber, showList).then((text) => {
//             e.target.innerHTML = text;
//         })
//     })
// })
//
// // 페이지 이동 이전 버튼 클릭
// prevButton.addEventListener("click", () => {
//     if (page > 1) {
// 		page -= 1;
//         adminNoticeService.getPagination(page, pageNumber, showList).then((text) => {
//             noticeBox.innerHTML = text;
//         })
//     }
// })
//
// // 페이지 이동 다음 버튼 클릭
// nextButton.addEventListener("click", () => {
//     if (page < `${pagination.page_count}`) {
//         page += 1;
//         adminNoticeService.getPagination(page, pageNumber, showList).then((text) => {
//             noticeBox.innerHTML = text;
//         })
//     }
// })

// 페이지 이동 막아주는 함수
function preventDefault(e) {
    e.preventDefault();
}

// 페이지 번호 클릭 시 페이지 이동 불가 이벤트 발생
pageNumberBox.addEventListener("click", (e) => {
    // 페이지 클릭 시 페이지 이동이 불가하도록 막기
    e.preventDefault()

    // 어떤 페이지 번호 눌렀는지 찾아오기
    // 막혀져 있는 페이지 박스 속 태그의 클래스명 0번째 인것을 가져오기
    const pageNumbers = e.target.classList[0]

    // 타겟의 첫번째 클래스 명이 'pagination'이라면,
    if (pageNumbers === 'pagination') {
        // pageNumbers 반복하여 pageNumber에 하나씩 담아주기
        page = e.target.textContent;
        console.log(page)
        // pageNumberButtons.forEach((pageNumberButton) => {
        //     pageNumberButton.
        //     adminNoticeService.getPagination(page, pageNumber, showList).then((text) => {
        //             e.target.innerHTML = text;
        //         })
        //
        // })

    } else if (pageNumbers === 'reft') {
        prevButton.addEventListener("click", () => {
            if (page > 1) {
                page -= 1;
                adminNoticeService.getPagination(page, pageNumber, showList).then((text) => {
                    noticeBox.innerHTML = text;
                })
            }
        })
    } else if (pageNumbers === 'right') {
        nextButton.addEventListener("click", () => {
            if (page < `${pagination.page_count}`) {
                page += 1;
                adminNoticeService.getPagination(page, pageNumber, showList).then((text) => {
                    noticeBox.innerHTML = text;
                })
            }
        })
    }
})



// ---------------------------------------------------------------------------------------------------------------------
// 검색
// 입력창
const searchInput = document.querySelector(".main-user-info-input")

// 전체보기 서치
searchInput.addEventListener('keyup', (e) => {
    if (e.keyCode === 13) {
        let keyword = e.target.value
        adminNoticeService.search(page, keyword, CreateService.showList).then((text) => {
            userData.innerHTML = text;
        })
        adminNoticeService.search(page, keyword, CreateService.showPaging).then((text) => {
            mainUserBottomUl.innerHTML = text;
        })
        adminNoticeService.search(page, keyword, CreateService.CountText).then((text) => {
            mainUserTotalNumber.textContent = text;
        })
    }
});



// // 페이지 번호 클릭 시 페이지 이동 불가 이벤트 발생
// pageNumberButtons.addEventListener("click", (e) => {
//     // 페이지 클릭 시 페이지 이동이 불가하도록 막기
//     e.preventDefault()
//
//     // 어떤 페이지 번호 눌렀는지 찾아오기
//     // 막혀져 있는 페이지 박스 속 태그의 클래스명 0번째 인것을 가져오기
//     const pageNumbers = document.querySelectorAll(`.${e.target.classList[0]}`)
//
//     // 타겟의 첫번째 클래스 명이 'pagination'이라면,
//     if (e.target.classList[0] === 'pagination') {
//         // pageNumbers 반복하여 pageNumber에 하나씩 담아주기
//         pageNumbers.forEach((pageNumber) => {
//             // pageNumber에 클릭 시 handlePageClick 함수 호출 이벤트 발생
//             pageNumber.addEventListener("click", handlePageClick);
//         })
//     }
// })
//
// // handlePageClick 함수 선언
// function handlePageClick(e) {
//     // 클릭된 페이지 번호 가져오기
//     const pageNumberQuery = e.target.querySelector(".main-user-number")
//     if (pageNumberQuery) {
//         // 가져온 번호의 텍스트를 page에 담기
//         const pageNumber = pageNumberQuery.textContent
//         page = pageNumber
//
//         // 페이지 클릭 시 해당하는 데이터 목록 가져오기
//         adminNoticeService.getPagination(page, pageNumber, showPageing).then((text) => {
//             // 가져온 데이터로 페이지 갱신
//             pagenumber.innerHTML = text;
//         });
//     }
// }