const allCheckedFn = () => {
    const checkedAllCategory = document.querySelector(".checked-all-category");
    const inputCheckboxes = document.querySelectorAll("input[type=checkbox]");

    checkedAllCategory.addEventListener("click", () => {
        for (const inputCheckbox of inputCheckboxes) {
            if (inputCheckbox.checked) {
                inputCheckbox.checked = false;
            } else {
                inputCheckbox.checked = true;
            }
        }
        insertCheckedMemberCount();
    });

}

const insertCheckedMemberCount = () => {
    const checkedMemberCount = document.querySelector(".checked-member-count");
    let inputCheckedboxes = document.querySelectorAll("input[type=checkbox]:checked");

    checkedMemberCount.innerText = `${inputCheckedboxes.length}명`;
};

const sendModalWrap = document.querySelector(".send-modal-wrap");
const sendToCheckedMember = document.querySelector(".send-to-checked-member");
const noneModalWrap = document.querySelector(".none-modal-wrap");

sendToCheckedMember.addEventListener("click", () => {
    let inputCheckedboxes = document.querySelectorAll("input[type=checkbox]:checked");

    if (inputCheckedboxes.length == 0) {
        noneModalWrap.querySelector(".none-modal-container").style.animation = "popUp 0.5s";
        noneModalWrap.style.display = "block";
        return;
    }

    for (let inputCheckedbox of inputCheckedboxes) {
        let email = inputCheckedbox.closest(".member-info-list").querySelector(".email");
        let name = inputCheckedbox.closest(".member-info-list").querySelector(".name");
        sendModalWrap.querySelector(".send-receiver-email").innerHTML += `<span class="email-span">${name.innerText}(${email.innerText})</span>`;
    }
    sendModalWrap.querySelector(".send-modal-container").style.animation = "popUp 0.5s";
    sendModalWrap.style.display = "block";
});

// none-modal-wrap의 닫기 버튼 클릭 시
const noneModalCloseBtn = document.querySelector(".none-modal-close-btn");

noneModalCloseBtn.addEventListener("click", () => {
    noneModalWrap.querySelector(".none-modal-container").style.animation = "popDown 0.5s";
    setTimeout(() => {
        noneModalWrap.style.display = "none";
    }, 450);
});

// 쪽지 작성 모달 내 textarea의 value 상태에 따라 보내기 버튼 활성화/비활성화
const textarea = sendModalWrap.querySelector("textarea[name=send-content]");
const sendCheckBtn = sendModalWrap.querySelector(".send-check-btn");

textarea.addEventListener("input", (e) => {
    sendCheckBtn.disabled = !e.target.value.trim();
});

sendCheckBtn.addEventListener('click', async () => {
    const sender = document.querySelector('.send-sender-email').innerText
    const content = document.querySelector('textarea').value
    const receiverElements = document.querySelectorAll('.email-span')
    let receivers = []
    receiverElements.forEach((element) => {
        receivers.push(element.innerText)
    })
    const letter = {
        'sender': sender,
        'receivers': receivers,
        'content': content
    }
    await mypageSendLetterService.post(letter)
})

// 쪽지 보내기 모달 버튼 클릭 시 발생하는 이벤트
const sendModalBtns = document.querySelectorAll(".send-modal-container button");

sendModalBtns.forEach((sendModalBtn) => {
    sendModalBtn.addEventListener("click", (e) => {
        if (e.target.className === "send-check-btn") {
            sendModalWrap.querySelector(".send-modal-container").style.animation = "popDown 0.5s";
            setTimeout(() => {
                sendModalWrap.querySelector(".send-modal-container").style.display = "none";
                sendModalWrap.querySelector(".check-modal-container").style.animation = "popUp 0.5s";
                sendModalWrap.querySelector(".check-modal-container").style.display = "flex";
                sendModalWrap.querySelector(".send-receiver-email").innerHTML = ``;
                sendModalWrap.querySelector("textarea[name=send-content]").value = ``;
                sendCheckBtn.disabled = true;
            }, 450);
        } else {
            sendModalWrap.querySelector(".send-modal-container").style.animation = "popDown 0.5s";
            setTimeout(() => {
                sendModalWrap.style.display = "none";
                sendModalWrap.querySelector(".send-receiver-email").innerHTML = ``;
                sendModalWrap.querySelector("textarea[name=send-content]").value = ``;
                sendCheckBtn.disabled = true;
            }, 450);
        }
    });
});

// 보내기 확인 모달 내 확인 클릭 시 모달 종료하는 이벤트
const checkModalCheckBtn = document.querySelector(".check-modal-container .check-btn");

checkModalCheckBtn.addEventListener("click", () => {
    sendModalWrap.querySelector(".check-modal-container").style.animation = "popDown 0.5s";
    setTimeout(() => {
        sendModalWrap.style.display = "none";
        sendModalWrap.querySelector(".check-modal-container").style.display = "none";
        sendModalWrap.querySelector(".send-modal-container").style.display = "flex";
    }, 450);
});

// 가입중 클릭 시 퇴출 확인 모달 나타나는 이벤트
const kickOutModalWrap = document.querySelector(".kick-out-modal-wrap");
const kickOutModalContainer = kickOutModalWrap.querySelector(".kick-out-modal-container");
let target;
let targetName;
const statusAddEvent = () => {
    const memberInfoDetails = document.querySelector(".member-info-details");

    memberInfoDetails.addEventListener("click", (e) => {
        if (e.target.closest('.member-status-join-btn')) {
            target = e.target.closest(".member-info-list");
            targetName = target.querySelector(".member-name").innerText;
            kickOutModalContainer.style.animation = "popUp 0.5s";
            kickOutModalContainer.querySelector(".kick-out-modal-title").innerText = `${targetName}님을 퇴출하시겠습니까?`;
            kickOutModalWrap.style.display = "block";
            const secondClass = e.target.closest('.member-info-list').classList[1]
            if (secondClass) {
                memberEmissionStatusUpdate(secondClass);
            }
        }

    });
};

statusAddEvent();

const memberEmissionStatusUpdate = (memberId) => {
    // 퇴출 모달 내 버튼 클릭 시 발생하는 이벤트
    const kickOutModalContainerBtns = kickOutModalWrap.querySelectorAll("button");
    const kickOutCheckModalContainer = kickOutModalWrap.querySelector(".kick-out-check-modal-container");

    kickOutModalContainerBtns.forEach((kickOutModalContainerBtn) => {
        kickOutModalContainerBtn.addEventListener("click", async (e) => {
            kickOutModalContainer.style.animation = "popDown 0.5s";
            if (e.target.className === "kick-out-btn") {
                // 데이터가 없어 임시 방편으로 사용
                if (memberId) {
                    const response = await mypageMemberStatusService.del(club_id, memberId)
                    if (response === 'ok') {
                        target.remove();
                    }
                }
                setTimeout(() => {
                    kickOutModalContainer.style.display = "none";
                    kickOutCheckModalContainer.querySelector(".modal-header-title").innerText = `${targetName}님을 퇴출했습니다.`;
                    kickOutCheckModalContainer.style.animation = "popUp 0.5s";
                    kickOutCheckModalContainer.style.display = "flex";
                }, 450);
            } else if (e.target.className === "kick-out-modal-cancle-btn") {
                setTimeout(() => {
                    kickOutModalWrap.style.display = "none";
                }, 450);
            } else {
                kickOutCheckModalContainer.style.animation = "popDown 0.5s";
                setTimeout(() => {
                    kickOutCheckModalContainer.style.removeProperty("animation");
                    kickOutCheckModalContainer.style.display = "none";
                    kickOutModalContainer.style.display = "flex";
                    kickOutModalWrap.style.display = "none";
                }, 450);
            }
        });
    });
}

// 가입대기 클릭 시 퇴출 확인 모달 나타나는 이벤트
const joinModalWrap = document.querySelector(".join-modal-wrap");
const joinModalContainer = joinModalWrap.querySelector(".join-modal-container");
const statusUpdateModal = () => {
    const memberStatusStandBtns = document.querySelectorAll(".member-status-stand-btn");

    memberStatusStandBtns.forEach((memberStatusStandBtn) => {
        memberStatusStandBtn.addEventListener("click", (e) => {
            target = e.target.closest(".member-info-list");
            targetName = target.querySelector(".member-name").innerText;

            joinModalContainer.style.animation = "popUp 0.5s";
            joinModalContainer.querySelector(".join-modal-title").innerText = `${targetName}님의 가입을 승인하시겠습니까?`;
            joinModalWrap.style.display = "block";
            const secondClass = e.target.closest('.member-info-list').classList[1]
            if (secondClass) {
                memberStatusJoinUpdate(secondClass);
            }
        });
    });
}


// 가입 모달 내 버튼 클릭 시 발생하는 이벤트

const memberStatusJoinUpdate = (memberId) => {
    const joinModalContainerBtns = joinModalWrap.querySelectorAll("button");
    const joinCheckModalContainer = joinModalWrap.querySelector(".join-check-modal-container");
    joinModalContainerBtns.forEach((joinModalContainerBtn) => {
        joinModalContainerBtn.addEventListener("click", async (e) => {
            joinModalContainer.style.animation = "popDown 0.5s";
            if (e.target.className === "join-btn") {
                // 데이터가 없어 임시 방편으로 사용
                if (memberId) {
                    const response = await mypageMemberStatusService.patch(club_id, memberId)
                    if (response === 'ok') {
                        target.querySelector(".member-status").innerHTML = `<div class="member-status-join-btn">가입중</div>`;
                    }
                }
                statusAddEvent();

                setTimeout(() => {
                    joinModalContainer.style.display = "none";
                    joinCheckModalContainer.querySelector(".modal-header-title").innerText = `${targetName}님의 가입을 승인했습니다.`;
                    joinCheckModalContainer.style.animation = "popUp 0.5s";
                    joinCheckModalContainer.style.display = "flex";
                }, 450);
            } else if (e.target.className === "join-modal-cancle-btn") {
                setTimeout(() => {
                    joinModalWrap.style.display = "none";
                }, 450);
            } else {
                joinCheckModalContainer.style.animation = "popDown 0.5s";
                setTimeout(() => {
                    joinCheckModalContainer.style.removeProperty("animation");
                    joinCheckModalContainer.style.display = "none";
                    joinModalContainer.style.display = "flex";
                    joinModalWrap.style.display = "none";
                }, 450);
            }
        });
    });
}
const addPaginationEvent = async (pageInfo) => {
    const prevBtn = document.querySelector(".prev-btn")
    const nextBtn = document.querySelector(".next-btn")
    const pageNumberBtn = document.querySelectorAll(".page-number-btn")
    const pageNumber = document.querySelectorAll(".page-number")

    prevBtn.addEventListener("click", async () => {
        if (pageInfo.currentPage <= 1) return;
        await mypageMemberListService.list(club_id, --page, order, search, showList);
    })

    nextBtn.addEventListener("click", async () => {
        if (pageInfo.currentPage === pageInfo.realEnd) return;
        await mypageMemberListService.list(club_id, ++page, order, search, showList);
    })

    pageNumberBtn.forEach((btn, i) => {
        btn.addEventListener("click", async () => {
            page = pageNumber[i].innerHTML;
            await mypageMemberListService.list(club_id, page, order, search, showList);
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
    if (orderList.length === 0) {
        text += ``
    } else {
        for (let item of orderList) {
            text += ``
            let statusHTML = '';
            if (item.status === 1) {
                statusHTML = `<div class="member-status"><div class="member-status-join-btn">가입중</div></div>`;
            } else if (item.status === -1) {
                statusHTML = `<div class="member-status"><div class="member-status-stand-btn">가입대기</div></div>`;
            } else {
                statusHTML = '';
            }
            text += `
            <div class="member-info-list ${item.id}">
                <div class="member-select-wrap">
                    <div>
                        <div class="member-select-container">
                            <div class="member-select-box">
                                <input type="checkbox" name="" id="member-checkbox"
                                       true-value="true" false-value="false"/>
                                <label class="member-checkbox-label" for="member-checkbox"></label>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="member-name"><div class="name">${item.member__member_nickname}</div></div>
                <div class="member-email">
                    <div class="email">${item.member__member_email}</div>
                </div>
                <div class="member-age-gender">
                    <div class="age">29/</div>
                    <div class="gender">
                     ${item.member__member_gender === 1 ? '남자' : item.member__member_gender === 2 ? '여자' : item.member__member_gender === 0 && '선택 안함'}
                    </div>
                </div>
                <div class="member-interest-area"><div class="address">${item.member__member_address}</div></div>
                <div class="member-interest-filed">
                    <div class="category">${item.member_favorite_categories[0] ? item.member_favorite_categories[0].category__category_name : ''}${item.category_count <= 0 || !item.category_count ? '' : '외 ' + item.category_count + '개'}</div>
                </div>
                ${statusHTML}
                <div class="member-join-date">
                    <div class="date">${item.member__created_date}</div>
                </div>
                <div class="member-activity">
                    <div>${item.activities[0] ? item.activities[0].activity__activity_title : ''}${item.activities.length === 0 ? '' : '외 ' + item.activities.length + '개'}</div>
                </div>
            </div>
        `
        }
    }
    memberInfoDetails.innerHTML = text
    showPagination(pageInfo)
    allCheckedFn()
    statusUpdateModal()
}
const memberStatusBox = document.querySelector('.member-status-box')
const memberSearchInput = document.querySelector('.member-search-input')
memberStatusBox.addEventListener('change', async (e) => {
    order = e.target.value
    page = 1
    await mypageMemberListService.list(club_id, page, order, search, showList)
});
memberSearchInput.addEventListener("keyup", async (e) => {
    search = e.target.value
    page = 1
    await mypageMemberListService.list(club_id, page, order, search, showList)
})

const memberInfoDetails = document.querySelector('.member-info-details');

memberInfoDetails.addEventListener('change', (e) => {
    if (e.target.matches('input[type=checkbox]')) {
        // 체크박스가 변경된 경우에만 실행되는 로직
        insertCheckedMemberCount();
    }
});
mypageMemberListService.list(club_id, page, order, search, showList);