// 전체선택 클릭 시 발생하는 이벤트
// 전체선택이라는 div를 가져온다

const checkedAllCategory = document.querySelector(".checked-all-category");
// 체크박스 전체를 가져온다
const inputCheckboxes = document.querySelectorAll("input[type=checkbox]");
page = 1
const inner = document.querySelector(".member-info-details")

checkedAllCategory.addEventListener("click", () => {
    // 체크된 모든 체크박스를 가져옵니다
    let inputCheckboxes = document.querySelectorAll("input[type=checkbox]");

    // 전체 체크 여부를 나타내는 변수를 초기화합니다
    let allChecked = true;

    // 각 체크박스를 확인하여 전체 체크 여부를 결정합니다
    inputCheckboxes.forEach((checkbox) => {
        if (!checkbox.checked) {
            // 만약 체크되지 않은 체크박스가 하나라도 있다면 전체 체크 여부를 false로 설정하고 체크를 해제합니다
            allChecked = false;
            checkbox.checked = true; // 체크박스를 체크합니다
        }
    });

    // 전체 체크 여부에 따라 처리합니다
    if (!allChecked) {
        // 전체 체크가 아니라면, 모든 체크박스를 체크합니다
        inputCheckboxes.forEach((checkbox) => {
            checkbox.checked = true;
        });
    } else {
        // 전체 체크라면, 모든 체크박스를 체크 해제합니다
        inputCheckboxes.forEach((checkbox) => {
            checkbox.checked = false;
        });
    }

    // 체크된 멤버 수를 삽입합니다
    insertCheckedMemberCount();
});





// input[type=checkbox]에 변화가 있을 경우 감지하고 insertCheckedMemberCount 함수를 실행해주는 이벤트
inputCheckboxes.forEach((inputCheckbox) => {
    inputCheckbox.addEventListener("change", () => {
        // 체크박스가 변경될 때마다 호출됩니다.
        insertCheckedMemberCount();
    });
});


// 체크된 개수를 선택된 구성원 수에 넣어주는 함수
const insertCheckedMemberCount = () => {
    const checkedMemberCount = document.querySelector(".checked-member-count");
    let inputCheckedboxes = document.querySelectorAll("input[type=checkbox]:checked");

    checkedMemberCount.innerText = `${inputCheckedboxes.length}명`;
};

// 쪽지 보내기 클릭 시 작성 모달 여는 이벤트
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
        sendModalWrap.querySelector(".send-receiver-email").innerHTML += `<span class="email-span">${email.innerText}</span>`;
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

// 쪽지 보내기 모달 버튼 클릭 시 발생하는 이벤트
// const sendModalBtns = document.querySelectorAll(".send-modal-container button");

// sendModalBtns.forEach((sendModalBtn) => {
//     sendModalBtn.addEventListener("click", (e) => {
//         if (e.target.className == "send-check-btn") {
//             sendModalWrap.querySelector(".send-modal-container").style.animation = "popDown 0.5s";
//             setTimeout(() => {
//                 sendModalWrap.querySelector(".send-modal-container").style.display = "none";
//                 sendModalWrap.querySelector(".check-modal-container").style.animation = "popUp 0.5s";
//                 sendModalWrap.querySelector(".check-modal-container").style.display = "flex";
//                 sendModalWrap.querySelector(".send-receiver-email").innerHTML = ``;
//                 sendModalWrap.querySelector("textarea[name=send-content]").value = ``;
//             }, 450);
//         } else {
//             sendModalWrap.querySelector(".send-modal-container").style.animation = "popDown 0.5s";
//             setTimeout(() => {
//                 sendModalWrap.style.display = "none";
//                 sendModalWrap.querySelector(".send-receiver-email").innerHTML = ``;
//                 sendModalWrap.querySelector("textarea[name=send-content]").value = ``;
//             }, 450);
//         }
//     });
// });

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

// 취소 버튼 클릭 시 퇴출 확인 모달 나타나는 이벤트
const kickOutModalWrap = document.querySelector(".kick-out-modal-wrap");
const kickOutModalContainer = kickOutModalWrap.querySelector(".kick-out-modal-container");
let target;
let targetName;



// 취소 모달 내 버튼 클릭 시 발생하는 이벤트
const kickOutModalContainerBtns = kickOutModalWrap.querySelectorAll("button");
const kickOutCheckModalContainer = kickOutModalWrap.querySelector(".kick-out-check-modal-container");


// 참가대기 클릭 시 참가 모달 나타나는 이벤트
const joinModalWrap = document.querySelector(".join-modal-wrap");
const joinModalContainer = joinModalWrap.querySelector(".join-modal-container");

const joinAddEvent = () => {
    let memberStatusStandBtns = document.querySelectorAll(".member-status-stand-btn");

    memberStatusStandBtns.forEach((memberStatusStandBtn) => {
        memberStatusStandBtn.addEventListener("click", (e) => {
            target = e.target.closest(".member-info-list");
            targetName = target.querySelector(".member-name").innerText;

            joinModalContainer.style.animation = "popUp 0.5s";
            joinModalContainer.querySelector(".join-modal-title").innerText = `${targetName}님의 참가를 승인하시겠습니까?`;
            joinModalWrap.style.display = "block";
        });
    });
};

joinAddEvent();

// 참가 모달 내 버튼 클릭 시 발생하는 이벤트
const joinModalContainerBtns = joinModalWrap.querySelectorAll("button");
const joinCheckModalContainer = joinModalWrap.querySelector(".join-check-modal-container");



// 참가중 클릭 시 대기 모달 나타나는 이벤트
const stopModalWrap = document.querySelector(".stop-modal-wrap");
const stopModalContainer = stopModalWrap.querySelector(".stop-modal-container");





// 대기 모달 내 버튼 클릭 시 발생하는 이벤트
const stopModalContainerBtns = stopModalWrap.querySelectorAll("button");
const stopCheckModalContainer = stopModalWrap.querySelector(".stop-check-modal-container");




const showList = async (member_list,clubManager)=>{
    // console.log(member_list)
    let text = '';
    console.log(clubManager);
    if (clubManager) {
       text += `<div class="member-info-list">
                    <div class="member-select-wrap">
                        <div>
                            <div class="member-select-container">
                                <div class="member-select-box">
                                    <input  style="display:none;" />
                                    <label ></label>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="member-name">${clubManager.club__member__member_nickname}</div>
                    <div class="member-email">
                        <div class="email">${clubManager.club__member__member_email}</div>
                    </div>
                    <div class="member-age-gender">
                        <div class="age">${calculateAge(clubManager.club__member__member_birth)}</div>
                        <div class="gender">${gender(clubManager.club__member__member_gender)}</div>
                    </div>
                    <div class="member-interest-filed">
                        <span>${getCategoryNames(clubManager.categories)}</span>
                    </div>
                    <div class="member-status">
                        <div class="member-status-join-btnaa">모임장</div>
                    </div>
                    <div class="member-join-date">
                        <div class="date">${changeDate(clubManager.created_date)}</div>
                    </div>
                    <div class="member-secessions">
                        <div class="member-secession-btn" style="display: none">활동취소</div>
                    </div>
                </div>`
    }
    member_list.forEach((member_list)=>{
        console.log(member_list)
        if(member_list.status === 1){
            text += `
            <div class="member-info-list">
                    <div class="member-select-wrap">
                        <div>
                            <div class="member-select-container">
                                <div class="member-select-box">
                                    <input type="checkbox" name="" id="member-checkbox" true-value="true" false-value="false" />
                                    <label class="member-checkbox-label" for="member-checkbox"></label>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="member-name">${member_list.member__member_nickname}</div>
                    <div class="member-email">
                        <div class="email">${member_list.member__member_email}</div>
                    </div>
                    <div class="member-age-gender">
                        <div class="age">${calculateAge(member_list.member__member_birth)}</div>
                        <div class="gender">${gender(member_list.member__member_gender)}</div>
                    </div>
                    <div class="member-interest-filed">
                        <span>${getCategoryNames(member_list.categories)}</span>
                    </div>
                    <div class="member-status">
                        <div class="member-status-join-btn ${member_list.id} ${member_list.member__member_nickname}">참가중</div>
                    </div>
                    <div class="member-join-date">
                        <div class="date">${changeDate(member_list.activity__created_date)}</div>
                    </div>
                    <div class="member-secessions">
                        <div class="member-secession-btn ${member_list.id} ${member_list.member__member_nickname}">취소</div>
                    </div>
                </div>`
        }else if(member_list.status === -1){
            text += `
            <div class="member-info-list">
                    <div class="member-select-wrap">
                        <div>
                            <div class="member-select-container">
                                <div class="member-select-box">
                                    <input type="checkbox" name="" id="member-checkbox" true-value="true" false-value="false" />
                                    <label class="member-checkbox-label" for="member-checkbox"></label>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="member-name">${member_list.member__member_nickname}</div>
                    <div class="member-email">
                        <div class="email">${member_list.member__member_email}</div>
                    </div>
                    <div class="member-age-gender">
                        <div class="age">${calculateAge(member_list.member__member_birth)}</div>
                        <div class="gender">${gender(member_list.member__member_gender)}</div>
                    </div>
                    <div class="member-interest-filed">
                        <span>${getCategoryNames(member_list.categories)}</span>
                    </div>
                    <div class="member-status">
                        <div class="member-status-stand-btn ${member_list.id} ${member_list.member__member_nickname}">참가대기</div>
                    </div>
                    <div class="member-join-date">
                        <div class="date">${changeDate(member_list.activity__created_date)}</div>
                    </div>
                    <div class="member-secessions">
                        <div class="member-secession-btn ${member_list.id} ${member_list.member__member_nickname}">취소</div>
                    </div>
                </div>`
        }
    })
    return text;
}

const cancle = document.querySelector(".join-modal-cancle-btn")
const cancleadd = document.querySelector(".stop-modal-cancle-btn")
const cansleaddd = document.querySelector(".kick-out-modal-cancle-btn")
const agree = document.querySelector(".join-btn")
const agrees = document.querySelector(".stop-btn")
const agreess = document.querySelector(".kick-out-btn")


inner.addEventListener("click", async (e)=>{
    const inputCheckboxes = document.querySelectorAll("input[type=checkbox]");
    inputCheckboxes.forEach((inputCheckbox) => {
    inputCheckbox.addEventListener("change", () => {
        // 체크박스가 변경될 때마다 호출됩니다.
        insertCheckedMemberCount();
    });
});

    if (e.target.classList[0] === 'member-status-stand-btn'){
        console.log('asd')
        let update = e.target.classList[1]
        let name = e.target.classList[2]
        joinModalContainer.style.animation = "popUp 0.5s";
        joinModalContainer.querySelector(".join-modal-title").innerText = `${name}님의 참가를 승인하시겠습니까?`;
        joinModalWrap.style.display = "block";
        cancle.addEventListener("click",async (e)=>{
            joinModalWrap.style.display = "none";
        })
        agree.addEventListener("click", async (e)=>{
            activityMemberService.update(update, activity_id)
            joinModalWrap.style.display = "none";
            await activityMemberService.getList(member_id, page,activity_id,member_status,search, showList).then((text) => {
            inner.innerHTML = text;
            updatePageButtons();
            insertCheckedMemberCount();
});
            await activityMemberService.getList(member_id, 1,activity_id,member_status,search,async (member_list,clubManager, total_pages) => {
            maxPage = total_pages;
            // 페이지 버튼 다시 업데이트
            updatePageButtons();
        });

        })


    }
    else if (e.target.classList[0] === 'member-status-join-btn'){
        let deletes = e.target.classList[1]
        let name = e.target.classList[2]
        stopModalContainer.style.animation = "popUp 0.5s";
        stopModalContainer.querySelector(".stop-modal-title").innerText = `${name}님의 상태를 변경하시겠습니까?`;
        stopModalWrap.style.display = "block";
        cancleadd.addEventListener("click",async (e)=>{
            stopModalWrap.style.display = "none";
        })
        agrees.addEventListener("click", async (e)=>{
            activityMemberService.remove(deletes, activity_id)
            stopModalWrap.style.display = "none";
            await activityMemberService.getList(member_id, page,activity_id,member_status,search, showList).then((text) => {
            inner.innerHTML = text;
            updatePageButtons();
            insertCheckedMemberCount();
});
            await activityMemberService.getList(member_id, 1,activity_id,member_status,search,async (member_list,clubManager, total_pages) => {
            maxPage = total_pages;
            // 페이지 버튼 다시 업데이트
            updatePageButtons();
        });

    })
    }
    else if (e.target.classList[0] === 'member-secession-btn'){
        let deletess = e.target.classList[1]
        let name = e.target.classList[2]
        kickOutModalContainer.style.animation = "popUp 0.5s";
        kickOutModalContainer.querySelector(".kick-out-modal-title").innerText = `${name}님의 활동 참여를 취소하시겠습니까?`;
        kickOutModalWrap.style.display = "block";
        cansleaddd.addEventListener("click",async (e)=>{
            kickOutModalWrap.style.display = "none";
        agreess.addEventListener("click", async (e)=>{
            activityMemberService.remove(deletess, activity_id)
            kickOutModalWrap.style.display = "none";
            await activityMemberService.getList(member_id, page,activity_id,member_status,search, showList).then((text) => {
            inner.innerHTML = text;
            updatePageButtons();
            insertCheckedMemberCount();
});
            await activityMemberService.getList(member_id, 1,activity_id,member_status,search,async (member_list,clubManager, total_pages) => {
            maxPage = total_pages;
            // 페이지 버튼 다시 업데이트
            updatePageButtons();
        });

    })
        })
    }

})



const conclebut = document.querySelector(".send-close-btn")

conclebut.addEventListener("click", ()=>{
    sendModalWrap.style.display = "none"
})

const send_message = document.querySelector(".send-check-btn")
send_message.addEventListener("click", async ()=>{
    let inputCheckedboxes = document.querySelectorAll("input[type=checkbox]:checked");
    const letter_content = sendModalWrap.querySelector("textarea[name=send-content]");
    for (let inputCheckedbox of inputCheckedboxes) {
        let email = inputCheckedbox.closest(".member-info-list").querySelector(".email");
        let receiver_email = `${email.innerText}`;
        await activityMemberService.write({
            letter_content : letter_content.value,
            receiver_email : receiver_email
        })
    }sendModalWrap.style.display = "none"

})














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
            const text = await activityMemberService.getList(member_id, currentPage,activity_id,member_status,search, showList);
            inner.innerHTML = text;

            pageButton.classList.add("focus-page");

            updatePageButtons(); // 페이지 버튼 업데이트
            insertCheckedMemberCount();
        });

        pageButtonsContainer.appendChild(pageButton);
    }
};

// 초기 페이지 로드
updatePageButtons();
insertCheckedMemberCount();

// 서버에서 전체 페이지 수를 가져와서 최대 페이지 업데이트
activityMemberService.getList(member_id, 1,activity_id,member_status,search,async (member_list,clubManager, total_pages) => {
    maxPage = total_pages;
    // 페이지 버튼 다시 업데이트
    updatePageButtons();
});

// 좌 버튼 클릭 이벤트 처리
leftButton.addEventListener('click', async () => {
    if (currentPage > 1) {
        currentPage--;
        const text = await activityMemberService.getList(member_id, currentPage,activity_id,member_status,search, showList);
        inner.innerHTML = text;

        updatePageButtons(); // 페이지 버튼 업데이트
        insertCheckedMemberCount();
    }
});

// 우 버튼 클릭 이벤트 처리
rightButton.addEventListener('click', async () => {
    if (currentPage < maxPage) {
        currentPage++;
        const text = await activityMemberService.getList(member_id, currentPage,activity_id,member_status,search, showList);
        inner.innerHTML = text;

        updatePageButtons(); // 페이지 버튼 업데이트
        insertCheckedMemberCount();
    }
});

// 초기 페이지 로드
updatePageButtons();

// 서버에서 전체 페이지 수를 가져와서 최대 페이지 업데이트
activityMemberService.getList(member_id, 1,activity_id,member_status,search,(member_list,clubManager, total_pages) => {
    maxPage = total_pages;
});











activityMemberService.getList(member_id, page,activity_id,member_status,search, showList).then((text) => {
    inner.innerHTML += text;
    insertCheckedMemberCount();
});




activityMemberService.getList(member_id, page,activity_id,member_status,search, showList);




const changeDate = (dateStr) => {
    const days = ['일', '월', '화', '수', '목', '금', '토'];
    const date = new Date(dateStr); // string타입을 date타입으로 바꿈.
    const month = date.getMonth() + 1; // getMonth()는 0부터 시작하므로 1을 더함.
    const day = date.getDate();
    const dayOfWeek = days[date.getDay()]; // 요일을 숫자로 가져와 해당 요일 문자열로 변환

    // padStart는 ,앞의 숫자의 자릿수 만큼 0을 채워줌
    return `${month.toString().padStart(2, '0')}월 ${day.toString().padStart(2, '0')}일(${dayOfWeek})`;
}

const gender = (value) => {
    if (value === 0) {
        return '/ 선택안함';
    } else if (value === 1) {
        return '/ 남성';
    } else if (value === 2) {
        return '/ 여성';
    }
};

const calculateAge = (birthYear) => {
    const currentYear = new Date().getFullYear();

    if (birthYear >= 1900 && birthYear <= currentYear) {
        const age = currentYear - birthYear;
        return `${age+1}`;
    } else {
        return '비공개';
    }
};


// 함수 정의
function getCategoryNames(categories) {
  const categoryIdToString = {
    1: '취미',
    2: '문화·예술',
    3: '운동 ·액티비티',
    4: '푸드·드링크',
    5: '여행·동행',
    6: '성장·자기개발',
    7: '동네·또래',
    8: '연애·소개팅',
    9: '재테크',
    10: '외국어',
    11: '스터디',
    12: '지역축제',
    13: '기타'
  };

  const categoryNames = [];
  categories.forEach(category => {
    const categoryId = category.category_id;
    const categoryName = categoryIdToString[categoryId];
    if (categoryName) {
      categoryNames.push(categoryName);
    }
  });

  const firstCategory = categoryNames.shift(); // 첫번째 카테고리를 저장하고 배열에서 제거
  const remainingCategoriesCount = categoryNames.length; // 남은 카테고리 개수

  if (remainingCategoriesCount === 0) {
    if (firstCategory) {
      return [firstCategory];
    } else {
      return ["관심분야 없음"];
    }
  } else {
    return [`${firstCategory} 외 ${remainingCategoriesCount}개`];
  }
}

