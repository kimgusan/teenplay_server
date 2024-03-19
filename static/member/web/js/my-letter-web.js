//  이메일 검사


const emailInput = document.getElementById('receiver-name');
let emailCheck = false;
emailInput.addEventListener("blur", async (e)=>{
    if (emailCheck) return;
    const resultMessage = document.getElementById('add-message');
    const email = emailInput.value;
    const message = await replyService.check(email)
    if (message.message === '존재하지 않는 이메일입니다.') {
        emailInput.style.border = '2px solid red'


        alert(message.message);
        emailCheck = false;
        return;
    }
    emailInput.value = message.message;
    emailCheck = true;
    emailInput.style.border = '1px solid rgba(0,0,0,.1)'

})




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



// 내용 클릭 시 해당 목록의 정보를 모달에 담고 보여주는 이벤트


// 작성하기 클릭 시 작성 모달 여는 이벤트
const sendModalWrap = document.querySelector(".send-modal-wrap");
const sendLetterBtn = document.querySelector(".send-letter-btn");

sendLetterBtn.addEventListener("click", () => {
    sendModalWrap.querySelector(".send-modal-container").style.animation = "popUp 0.5s";
    sendModalWrap.style.display = "block";
});

// 받는 사람과 내용 value 상태에 따라 활성화/비활성화
const sendReceiverEmail = sendModalWrap.querySelector(".send-receiver-email");
const textarea = sendModalWrap.querySelector("textarea[name=send-content]");
const sendCheckBtn = sendModalWrap.querySelector(".send-check-btn");

const countValue = () => {
    if (emailCheck === true && textarea.value) {
        sendCheckBtn.disabled = false;
    } else {
        sendCheckBtn.disabled = true;
    }
};

countValue();

sendReceiverEmail.addEventListener("input", () => {
    countValue();
});

textarea.addEventListener("input", () => {
    countValue();
});

// 쪽지 보내기 모달 버튼 클릭 시 발생하는 이벤트
const sendModalBtns = document.querySelectorAll(".send-modal-container button");
sendModalBtns.forEach((sendModalBtn) => {
    sendModalBtn.addEventListener("click", (e) => {
        if (e.target.className == "send-check-btn") {
            sendModalWrap.querySelector(".send-modal-container").style.animation = "popDown 0.5s";
            setTimeout(() => {
                sendModalWrap.querySelector(".send-modal-container").style.display = "none";
                sendModalWrap.querySelector(".check-modal-container").style.animation = "popUp 0.5s";
                sendModalWrap.querySelector(".check-modal-container").style.display = "flex";
                sendReceiverEmail.value = ``;
                textarea.value = ``;
                countValue();
            }, 450);
        } else {
            sendModalWrap.querySelector(".send-modal-container").style.animation = "popDown 0.5s";
            setTimeout(() => {
                sendModalWrap.style.display = "none";
                sendReceiverEmail.value = ``;
                textarea.value = ``;
                countValue();
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

let page = 1
const tbody = document.querySelector(".add-more");



const writeButton = document.getElementById("send-check-button")

writeButton.addEventListener("click", async (e) => {
    const letter_content = document.getElementById("letter-content");
    const receiver_id = document.getElementById("receiver-name")
    await replyService.write({
        letter_content: letter_content.value,
        receiver_id: receiver_id.value
    });
    await replyService.getList(member_id, page,status_letter, showList).then((text) => {
    tbody.innerHTML = text;
    });
    await replyService.getList(member_id, 1,status_letter,(replies, total_pages) => {
    maxPage = total_pages;
    });

    updatePageButtons();
})


replyService.getList(member_id, 1, status_letter,(replies, total_pages) => {
maxPage = total_pages;
});


const showList = async (replies) => {
    let text = '';
    if (replies.length === 0 ){
       text += `<div class="test" style="    padding-top: 22px;
    padding-left: 482px;
    padding-bottom: 36px;">아직 작성한 쪽지가 없습니다.</div>`
    }
    replies.forEach((letter) =>  {
        if (replies.length ===0) {
            `<div className="test">아직 주고 받은 쪽지가 없습니다.</div>`
        } else {
            text += `
            <tr class="letter-details">
                <td class="letter-content">
                    <a class="letter-text ${letter.id}" href="#" data-letter-content="${letter.letter_content}">${letter.letter_content}</a>
                </td>
                <td class="letter-sender">${letter.sender__member_nickname}</td>
                <td class="letter-receiver">${letter.receiver__member_nickname}</td>
                <td class="letter-write-time">${changeDate(letter.created_date)}</td>
                <td>
                    <button class="letter-remove ${letter.id}">삭제하기</button>
                </td>
            </tr>
        `}

    });
    return text;

};





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
            const text = await replyService.getList(member_id, currentPage, status_letter, showList);
            tbody.innerHTML = text;

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
replyService.getList(member_id, 1, status_letter, async (replies, total_pages) => {
    maxPage = total_pages;
    // 페이지 버튼 다시 업데이트
    updatePageButtons();
});

// 좌 버튼 클릭 이벤트 처리
leftButton.addEventListener('click', async () => {
    if (currentPage > 1) {
        currentPage--;
        const text = await replyService.getList(member_id, currentPage, status_letter, showList);
        tbody.innerHTML = text;

        updatePageButtons(); // 페이지 버튼 업데이트
    }
});

// 우 버튼 클릭 이벤트 처리
rightButton.addEventListener('click', async () => {
    if (currentPage < maxPage) {
        currentPage++;
        const text = await replyService.getList(member_id, currentPage, status_letter, showList);
        tbody.innerHTML = text;

        updatePageButtons(); // 페이지 버튼 업데이트
    }
});


// 서버에서 전체 페이지 수를 가져와서 최대 페이지 업데이트
replyService.getList(member_id, 1,status_letter,(replies, total_pages) => {
    maxPage = total_pages;
});






const deleteModalwrap = document.querySelector(".delete-modal-wrap");
const deleteBut = document.getElementById('delete-but')
const deleteButclosed = document.getElementById('delete-out')
const letterModalwrap = document.querySelector(".letter-modal-wrap");


tbody.addEventListener("click", async (e)=>{
    if(e.target.classList[0]==='letter-remove'){
        const letter_id = e.target.classList[1]
        deleteModalwrap.style.display = 'block'
        deleteBut.addEventListener("click",async (e)=>{
            await replyService.remove(letter_id);
            await replyService.getList(member_id, page,status_letter, showList).then((text) => {
                tbody.innerHTML = text;
            });
            await replyService.getList(member_id, 1,status_letter,(replies, total_pages) => {
            maxPage = total_pages;
            });


            page = 1
            currentPage = 1
            deleteModalwrap.style.display = 'none'
            updatePageButtons();
        })
    }else if(e.target.classList[0] === 'letter-text'){


            const letterTextBtns = document.querySelectorAll(".letter-text");

            let letterDetails;

            letterTextBtns.forEach((letterTextBtn) => {
                letterTextBtn.addEventListener("click", (e) => {
                    letterDetails = e.target.closest(".letter-details");
                    // letter-details
                    letterModalwrap.querySelector(".sender-email").innerText = letterDetails.querySelector(".letter-sender").innerText;
                    letterModalwrap.querySelector(".receiver-email").innerText = letterDetails.querySelector(".letter-receiver").innerText;
                    letterModalwrap.querySelector(".write-time").innerText = letterDetails.querySelector(".letter-write-time").innerText;
                    letterModalwrap.querySelector("textarea[name=letter-content]").value = e.target.innerText;
                    letterModalwrap.querySelector(".letter-modal-container").style.animation = "popUp 0.5s";
                    letterModalwrap.style.display = "block";

                });
            })

            // 상세보기 모달 닫기 클릭 시 모달을 끄는 이벤트
            const letterModalCloseBtn = document.querySelector(".letter-modal-wrap .close-btn");

            letterModalCloseBtn.addEventListener("click", () => {
                letterModalwrap.querySelector(".letter-modal-container").style.animation = "popDown 0.5s";
                setTimeout(() => {
                    letterModalwrap.style.display = "none";
                }, 450);
            });
            ;
            const letter_id = e.target.classList[1]
            await replyService.update(letter_id);
            updatePageButtons();

    }
})
deleteButclosed.addEventListener("click", async (e) =>{
    deleteModalwrap.style.display = 'none'
})






// "add all" 버튼 클릭 이벤트 처리


replyService.getList(member_id, page,status_letter, showList).then((text) => {
    tbody.innerHTML += text;
});




replyService.getList(member_id, page,status_letter, showList);



const changeDate = (dateStr) => {
    const days = ['일', '월', '화', '수', '목', '금', '토'];
    const date = new Date(dateStr); // string타입을 date타입으로 바꿈.
    const month = date.getMonth() + 1; // getMonth()는 0부터 시작하므로 1을 더함.
    const day = date.getDate();
    const dayOfWeek = days[date.getDay()]; // 요일을 숫자로 가져와 해당 요일 문자열로 변환

    // padStart는 ,앞의 숫자의 자릿수 만큼 0을 채워줌
    return `${month.toString().padStart(2, '0')}월 ${day.toString().padStart(2, '0')}일(${dayOfWeek})`;
}
