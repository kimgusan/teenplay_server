// 정렬 우선 순위 선택하는 버튼 클릭 시 모달 열기
const tabListBtn = document.querySelector(".tab-list-btn");
const tabList = document.querySelector(".tab-list");
let text = ``;
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
    tab.addEventListener("click", async () => {
        tabListBtn.firstElementChild.innerText = tab.innerText;
        order = tabListBtn.firstElementChild.innerText
        text = ``;
        await mypageClubListService.list(page, order, showList)
    });
});

const alarmStatusHandler = () => {
    const signalBtns = document.querySelectorAll("#signal-btn");
    signalBtns.forEach((signalBtn) => {
        signalBtn.addEventListener("click", async (e) => {
            let signalStatus = e.target.closest("#signal-btn").querySelector("span");
            let messageModalBox = document.querySelector(".message-modal-box");

            messageModalBox.style.animation = "popUp 0.5s";

            if (signalStatus.innerText === "알림 받는 중") {

                const clubId = e.target.closest('.club-box').classList[1]
                await mypageClubAlarmStatusService.alarm(clubId)

                signalStatus.innerText = "알림 설정";

                e.target.closest("#signal-btn").className = "signal-off";
                e.target.closest("#signal-btn").querySelector(".signal-on-svg").classList.replace("signal-on-svg", "signal-off-svg");
                const messageMaodalContainer = document.querySelector(".message-maodal-container");
                messageModalBox.querySelector(".message-check-box").style.display = "none";
                messageModalBox.querySelector(".modal-header-title").innerText = "모임 알림을 해제했습니다.";
                messageModalBox.querySelector(".message-guide-ment").innerText = "더 이상 새로운 모임 알림을 받을 수 없습니다.";
                messageModalBox.querySelector(".continuously-btn").innerText = "확인";
                messageModalBox.querySelector(".club-page-btn").style.display = "none";
                messageMaodalContainer.style.display = "block";
                messageModalBox.querySelector(".club-page-btn").addEventListener('click', () => {
                    window.location = `/member/mypage-club/${clubId}`
                })


            } else if (signalStatus.innerText === "알림 설정") {

                const clubId = e.target.closest('.club-box').classList[1]
                await mypageClubAlarmStatusService.alarm(clubId)
                signalStatus.innerText = "알림 받는 중";
                e.target.closest("#signal-btn").className = "signal-on";
                e.target.closest("#signal-btn").querySelector(".signal-off-svg").classList.replace("signal-off-svg", "signal-on-svg");
                const messageMaodalContainer = document.querySelector(".message-maodal-container");
                messageModalBox.querySelector(".message-check-box").style.display = "";
                messageModalBox.querySelector(".modal-header-title").innerText = "모임 알림을 설정했습니다.";
                messageModalBox.querySelector(".message-guide-ment").innerText = "모임에 새로운 행사가 개설되면 알림으로 알려드려요";
                messageModalBox.querySelector(".continuously-btn").innerText = "계속 살펴보기";
                messageModalBox.querySelector(".club-page-btn").style.display = "";
                messageMaodalContainer.style.display = "block";
            }

        });
    });
}


const modalBtnHandler = () => {
    const messageModalBox = document.querySelector(".message-modal-box");

    messageModalBox.addEventListener("click", (e) => {
        const messageMaodalContainer = document.querySelector(".message-maodal-container");
        if (e.target.closest('.continuously-btn')) {
            messageModalBox.style.animation = "popDown 0.5s";
            setTimeout(() => {
                messageMaodalContainer.style.display = "none";
            }, 450);
        } else {
            // 해당 모임 상세페이지 주소로 이동
            messageMaodalContainer.style.display = "none";
        }
    });
}
const addPaginationEvent = async () => {
    const nextBtn = document.querySelector(".teenchin-more-btn")

    nextBtn.addEventListener("click", async () => {
        await mypageClubListService.list(++page, order, showList);
    })
}
const moreBtnBox = document.querySelector('.more-btn-box')
const showPagination = (orderList) => {
    console.log(orderList)
    if (orderList.length === 0) {
        moreBtnBox.innerHTML = ``;

        return;
    }
    moreBtnBox.innerHTML = `
        <button class="teenchin-more-btn" type="button">
            <div class="more-text-box">
                <div style="margin-right: 0.5rem">더보기</div>
                <svg xmlns="http://www.w3.org/2000/svg" class="down-arrow-svg" fill="none"
                     viewBox="0 0 24 24" stroke="currentColor" stroke-width="3">
                    <path stroke-linecap="round" stroke-linejoin="round"
                          d="M19 9l-7 7-7-7"></path>
                </svg>
            </div>
        </button>
        `;
    addPaginationEvent();
}


const clubList = document.querySelector('.club-list')

const showList = (orderList) => {
    if (orderList.length === 0) {
        text += `<div class="club-none-box" style="margin-top: 30px">
                                <div class="club-none-ment">가입한 모임이 없습니다.</div>
                                <!-- 모임 홍보 게시판 주소 필요 -->
                                <a class="club-pr-link" href=""> 모임 가입하러 가기 </a>
                            </div>`
    } else {
        for (let club of orderList) {
            text += `
            <div class="club-box ${club.club_id}">
                <div class="club-items">
                    <!-- 모임 상세보기 이동 주소 필요 -->
                    <a href="${club.join_status === 2 ? '/member/mypage-club/' + club.club_id : '/club/detail/?id=' + club.club_id}">
                        <div class="club-profile-img-contents">
                            <div class="club-profile-img-box">
                                <img class="club-profile-img" src="/upload/${club.profile_path}" />
                            </div>
                            <div class="club-profile-img-gap"></div>
                        </div>
                        <div class="club-info-contents">
                            <div class="club-name">${club.name}</div>
                            <div class="club-info-item">${club.activity_count} <span>개의 활동</span></div>
                        </div>
                    </a>
                    <div class="club-btn-container">
                        <div class="club-btn-box">
                            ${generateButtonHTML(club.join_status, club.alarms, club.club_id)}
                        </div>
                    </div>
                </div>
            </div>
        `
        }
    }
    clubList.innerHTML = text;
    showPagination(orderList)
    modalBtnHandler()
    alarmStatusHandler()
}

const generateButtonHTML = (joinStatus, alarms, club_id) => {
    if (joinStatus === 2) {
        return `
            <a class="management-btn" href="/member/mypage-club/${club_id}">
                <svg viewBox="0 0 24 24" fill="rgb(36 93 203/var(--tw-text-opacity))"
                     preserveAspectRatio="xMidYMid meet" class="management-svg" focusable="false"
                     style="pointer-events: none">
                    <g>
                        <path
                            d="M22.7 19l-9.1-9.1c.9-2.3.4-5-1.5-6.9-2-2-5-2.4-7.4-1.3L9 6 6 9 1.6 4.7C.4 7.1.9 10.1 2.9 12.1c1.9 1.9 4.6 2.4 6.9 1.5l9.1 9.1c.4.4 1 .4 1.4 0l2.3-2.3c.5-.4.5-1.1.1-1.4z"></path>
                    </g>
                </svg>
                <span>관리하기</span>
            </a>`;
    } else if (joinStatus === -1) {
        return `
            <button class="club-wait-btn" type="button" disabled>
                <svg class="check-svg" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2C17.523 2 22 6.478 22 12C22 17.522 17.523 22 12 22C6.477 22 2 17.522 2 12C2 6.478 6.477 2 12 2ZM12 3.667C7.405 3.667 3.667 7.405 3.667 12C3.667 16.595 7.405 20.333 12 20.333C16.595 20.333 20.333 16.595 20.333 12C20.333 7.405 16.595 3.667 12 3.667ZM11.9987 14.5022C12.5502 14.5022 12.9973 14.9494 12.9973 15.5009C12.9973 16.0524 12.5502 16.4996 11.9987 16.4996C11.4471 16.4996 11 16.0524 11 15.5009C11 14.9494 11.4471 14.5022 11.9987 14.5022ZM11.9945 7C12.3742 6.9997 12.6882 7.2816 12.7381 7.64764L12.7451 7.7494L12.7487 12.251C12.749 12.6652 12.4135 13.0013 11.9993 13.0016C11.6196 13.0019 11.3055 12.72 11.2556 12.354L11.2487 12.2522L11.2451 7.7506C11.2447 7.33639 11.5802 7.00033 11.9945 7Z"></path>
                </svg>
                <span>가입대기</span>
            </button>`;
    } else if (alarms) {
        return `
            <button class="club-btn" type="button" disabled>
                <svg xmlns="http://www.w3.org/2000/svg" class="check-svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd"
                          d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                          clip-rule="evenodd"></path>
                </svg>
                <span>가입중</span>
            </button>
            <button id="signal-btn" class="signal-on" type="button">
                <svg xmlns="http://www.w3.org/2000/svg" class="signal-on-svg" viewBox="0 0 20 20"
                     fill="currentColor">
                    <path
                        d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6zM10 18a3 3 0 01-3-3h6a3 3 0 01-3 3z"></path>
                </svg>
                <span>알림 받는 중</span>
            </button>`;
    }
}
mypageClubListService.list(page, order, showList)
