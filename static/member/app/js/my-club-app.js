// 정렬 우선 순위 선택하는 버튼 클릭 시 모달 열기
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

// 알림 버튼 클릭 시 알림 상태에 따라 다른 모달이 나오는 이벤트
const signalBtns = document.querySelectorAll("#signal-btn");
const messageMaodalContainer = document.querySelector(".message-maodal-container");
const messageModalBox = document.querySelector(".message-modal-box");

signalBtns.forEach((signalBtn) => {
    signalBtn.addEventListener("click", (e) => {
        let signalStatus = e.target.closest("#signal-btn").querySelector("span");
        messageModalBox.style.animation = "popUp 0.5s";

        if (signalStatus.innerText == "알림 받는 중") {
            signalStatus.innerText = "알림 설정";

            e.target.closest("#signal-btn").className = "signal-off";
            e.target.closest("#signal-btn").querySelector(".signal-on-svg").classList.replace("signal-on-svg", "signal-off-svg");

            messageModalBox.querySelector(".message-check-box").style.display = "none";
            messageModalBox.querySelector(".modal-header-title").innerText = "모임 알림을 해제했습니다.";
            messageModalBox.querySelector(".message-guide-ment").innerText = "더 이상 새로운 모임 알림을 받을 수 없습니다.";
            messageModalBox.querySelector(".continuously-btn").innerText = "확인";
            messageModalBox.querySelector(".club-page-btn").style.display = "none";

            messageMaodalContainer.style.display = "block";
            return;
        }
        signalStatus.innerText = "알림 받는 중";
        e.target.closest("#signal-btn").className = "signal-on";
        e.target.closest("#signal-btn").querySelector(".signal-off-svg").classList.replace("signal-off-svg", "signal-on-svg");
        messageModalBox.querySelector(".message-check-box").style.display = "";
        messageModalBox.querySelector(".modal-header-title").innerText = "모임 알림을 설정했습니다.";
        messageModalBox.querySelector(".message-guide-ment").innerText = "모임에 새로운 행사가 개설되면 메일로 알려드려요";
        messageModalBox.querySelector(".continuously-btn").innerText = "계속 살펴보기";
        messageModalBox.querySelector(".club-page-btn").style.display = "";
        messageMaodalContainer.style.display = "block";
    });
});

const moveBtns = document.querySelectorAll(".move-btn-box button");

moveBtns.forEach((moveBtn) => {
    moveBtn.addEventListener("click", () => {
        if (moveBtn.className == "continuously-btn") {
            messageModalBox.style.animation = "popDown 0.5s";
            setTimeout(() => {
                messageMaodalContainer.style.display = "none";
            }, 450);
        } else {
            // 해당 모임 상세페이지 주소로 이동
            messageMaodalContainer.style.display = "none";
        }
    });
});
