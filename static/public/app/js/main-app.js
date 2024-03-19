const slideBannerBox = document.querySelector(".slide-banner-box");
const bannerPageCount = document.querySelector(".banner-page-count");
let count = 0;

// 첫 화면에서는 첫번째 배너가 보이지만 이 함수가 실행되는 순간 이제 두 번째 배너로 넘어가는 것이다.
function autoSlide() {
    // 이동되는 데 걸리는 시간은 0.5초
    slideBannerBox.style.transition = "transform 0.5s";
    // 마지막 슬라이드일 때
    // 3번 뒤에 1번 배치시킨다.
    // 3번에서 1번으로 슬라이드 진행
    // 0s를 줘서 원래 1번 위치로 이동(슬라이드 효과는 안보임)
    count++;
    if (count == 3) {
        slideBannerBox.style.transform = "translate(-" + 100 * (count + 1) + "%)";
        setTimeout(() => {
            slideBannerBox.style.transition = "transform 0s";
            slideBannerBox.style.transform = "translate(-100%)";
        }, 500);
        count = 0;
        bannerPageCount.innerText = `${count + 1}/3`;
    } else {
        // 처음에는 첫번째 배너가 선택되어 있기 때문에 다음으로 넘어가면서 이전 배너부분의 버튼 색을 돌려준다.
        bannerPageCount.innerText = `${count + 1}/3`;

        // 왼쪽으로 -100 * (count + 1) 만큼 이동한다.
        // 왜 count에 + 1을 한 것인가? : 가장 앞에 6번 배너부터 시작되기 때문이다.
        slideBannerBox.style.transform = "translate(-" + 100 * (count + 1) + "%)";
    }
}

// 무한 반복
let firstDiv = document.createElement("div");
firstDiv.classList.add(`slide-content-wrap`);
let lastDiv = document.createElement("div");
lastDiv.classList.add(`slide-content-wrap`);

// 가장 마지막에 첫번째 배너를 이어 붙인다, 슬라이드 모션이 자연스럽게 1번으로 돌아가게 하기 위함
let firstText = `<div class="slide-content-container">
                    <img class="slide-content-background" alt=" KYOBO x 창업도약패키지" src="/staticfiles/images/main/app/main-app-banner01.png" />
                    <div class="slide-text-content">
                        <!-- 해당 축제 링크 필요 -->
                        <a class="slide-text-link" href="" target="_blank" ondragstart="return false">
                            <div class="slide-text-box">
                                <div>
                                    <div class="festival-title">KYOBO x 창업도약패키지</div>
                                    <div class="festival-contents">
                                        사업설명회 : 24. 2. 19.(월) 15:00 / 광화문 교보생명빌딩 17층 이노스테이지
                                        <br />
                                        협업분야 : 보험·AI
                                    </div>
                                </div>
                                <div class="festival-info">
                                    <div>2024. 01. 30.(화) ~ 2024. 02. 23.(금)</div>
                                    <div>온라인 접수</div>
                                </div>
                            </div>
                        </a>
                    </div>
                </div>`;

firstDiv.innerHTML = firstText;
slideBannerBox.appendChild(firstDiv);

// 가장 첫번째에 마지막 배너를 이어 붙인다, 이전 버튼 클릭 시 슬라이드 모션이 자연스럽게 6번으로 돌아가게 하기 위함
let lastText = `<div class="slide-content-container">
                    <img class="slide-content-background" alt="가장 효과적인 생존전략 d·camp officehours" src="/staticfiles/images/main/app/main-app-banner03.png" />
                    <div class="slide-text-content">
                        <!-- 해당 축제 링크 필요 -->
                        <a class="slide-text-link" href="" target="_blank" ondragstart="return false">
                            <div class="slide-text-box" style="color: rgb(53, 37, 40)">
                                <div>
                                    <div class="festival-title">
                                        가장 효과적인 생존전략
                                        <br />
                                        d·camp officehours
                                    </div>
                                    <div class="festival-contents">
                                        스타트업 생존 패키지,
                                        <br />
                                        투자 검토는 물론 사업 고도화에 필요한 라인업을 만나보세요!
                                    </div>
                                </div>
                                <div class="festival-info">
                                    <div>2024. 02. 20.(화) ~ 2024. 02. 29.(목)</div>
                                    <div>오프라인 행사</div>
                                </div>
                            </div>
                        </a>
                    </div>
                </div>`;

lastDiv.innerHTML = lastText;
slideBannerBox.insertBefore(lastDiv, document.querySelector("div.slide-banner-box div"));

// 첫번째 버튼이 무조건 첫번째 배너이기 때문에 검은색 칠하고 시작
bannerPageCount.innerText = `${count + 1}/3`;

// 첫번째 배너는 3번이니까 왼쪽으로 한 번 밀어서 1번 보이게 함.
slideBannerBox.style.transform = "translate(-100%)";

// 4초마다 슬라이드 이동
let inter = setInterval(autoSlide, 4000);

// 베너 페이지 수 옆 + 클릭 시 베너 모달창 block처리 이벤트
const bannerMoreButton = document.querySelector(".banner-more-button");
const bannerModalContainer = document.querySelector(".banner-modal-container");

bannerMoreButton.addEventListener("click", () => {
    bannerModalContainer.style.display = "block";
});

const bannerCloseContainer = document.querySelector(".banner-close-container");

bannerCloseContainer.addEventListener("click", () => {
    bannerModalContainer.style.display = "none";
});

const bannerModalBox = document.querySelector(".banner-modal-box");

bannerModalBox.addEventListener("click", () => {
    bannerModalContainer.style.display = "none";
});

// 좋아요 버튼 클릭 시 svg 변경되는 이벤트
HTMLCollection.prototype.forEach = Array.prototype.forEach;
const likeBtns = document.querySelectorAll(".like-btn");

likeBtns.forEach((likeBtn) => {
    likeBtn.addEventListener("click", () => {
        let spans = likeBtn.children;
        spans.forEach((span) => {
            if (span.style.display) {
                span.style.display === "none" ? (span.style.display = "block") : (span.style.display = "none");
                return;
            }
            span.style.display = "none";
        });
    });
});

// 가입 신청 클릭 시 모달, 가입상태, 알림을 표시하는 이벤트
const joinBtns = document.querySelectorAll(".join-btn");
const joinMaodalContainer = document.querySelector(".join-maodal-container");
const joinGuideMent = document.querySelector(".join-guide-ment");
const joinApplicationMent = document.querySelector(".join-application-ment");
const joinCheckBox = document.querySelector(".join-check-box");

joinBtns.forEach((joinBtn) => {
    joinBtn.addEventListener("click", () => {
        joinBtn.style.display = "none";
        joinBtn.previousElementSibling.style.display = "flex";
        joinCheckBox.style.display = "flex";
        joinMaodalContainer.style.display = "block";
        joinCancleMent.style.display = "none";
        joinApplicationMent.style.display = "block";
        joinGuideMent.innerText = "가입 신청이 처리 된다면 알림으로 알려드려요.";
        continuouslyBtn.style.removeProperty("flex");
        continuouslyBtn.innerText = "계속 살펴보기";
        clupPageBtn.style.display = "flex";
        clupPageBtn.innerText = "모임 상세 바로가기";
    });
});

const loginMent = document.querySelector(".login-ment");

// 모임 가입 버튼 클릭 시 로그인 상태가 아니라면 나오는 모달 내용
// joinBtns.forEach((joinBtn) => {
//     joinBtn.addEventListener("click", () => {
//         joinMaodalContainer.style.display = "block";
//         joinCheckBox.style.display = "none";
//         loginMent.style.display = "block";
//         joinGuideMent.innerText = "바로 로그인하고, 모임을 가입하세요.\n새롭게 개설되는 활동 안내를 받을 수 있습니다.";
//         continuouslyBtn.style.removeProperty("flex");
//         continuouslyBtn.innerText = "취소";
//         clupPageBtn.style.display = "flex";
//         clupPageBtn.innerText = "로그인 하기";
//     });
// });

// 계속 살펴보기 또는 확인 클릭 시 모달을 닫는 이벤트
const continuouslyBtn = document.querySelector(".continuously-btn");

continuouslyBtn.addEventListener("click", () => {
    joinMaodalContainer.style.display = "none";
    joinCancleMent.style.display = "none";
    joinApplicationMent.style.display = "none";
    signalCancleMent.style.display = "none";
    loginMent.style.display = "none";
});

// 가입중 클릭 시 모달, 가입 신청을 표기하는 이벤트
const joinStatuses = document.querySelectorAll(".join-status");
const joinCancleMent = document.querySelector(".join-cancle-ment");
const clupPageBtn = document.querySelector(".clup-page-btn");

joinStatuses.forEach((joinStatus) => {
    joinStatus.addEventListener("click", () => {
        joinMaodalContainer.style.display = "block";
        joinCheckBox.style.display = "none";
        joinCancleMent.style.display = "block";
        joinGuideMent.innerText = "모임에서 탈퇴했습니다.\n더 이상 새로운 모임 알림을 받을 수 없습니다.";
        joinStatus.parentElement.style.display = "none";
        joinStatus.parentElement.nextElementSibling.style.display = "flex";
        continuouslyBtn.style.flex = "1 1 0%";
        continuouslyBtn.innerText = "확인";
        clupPageBtn.style.display = "none";
    });
});

// 알림 받는 중 버튼 클릭 시 모달, 제목, 내용 등 바꾸는 이벤트
const signalBtns = document.querySelectorAll("#signal-btn");
const signalCancleMent = document.querySelector(".signal-ment");
let signalCheck = true;

signalBtns.forEach((signalBtn) => {
    signalBtn.addEventListener("click", () => {
        signalBtn.classList.toggle("signal-on");
        signalBtn.classList.toggle("signal-off");
        signalBtn.firstElementChild.classList.toggle("signal-on-svg");
        signalBtn.firstElementChild.classList.toggle("signal-off-svg");

        let spanText = signalBtn.lastElementChild.innerText;
        if (spanText === "알림 받는 중") {
            signalBtn.lastElementChild.innerText = "알림 설정";
        } else {
            signalBtn.lastElementChild.innerText = "알림 받는 중";
        }

        joinMaodalContainer.style.display = "block";

        if (signalCheck) {
            joinCheckBox.style.display = "none";
            signalCancleMent.style.display = "block";
            signalCancleMent.innerText = "모임알림을 해제했습니다.";
            joinGuideMent.innerText = "더 이상 새로운 모임 알림을 받을 수 없습니다.";
            continuouslyBtn.style.flex = "1 1 0%";
            continuouslyBtn.innerText = "확인";
            clupPageBtn.style.display = "none";
            signalCheck = false;
        } else {
            joinCheckBox.style.display = "flex";
            signalCancleMent.style.display = "block";
            signalCancleMent.innerText = "모임알림을 설정했습니다.";
            joinGuideMent.innerText = `모임에 새로운 소식이 있다면 알림으로 알려드려요.`;
            continuouslyBtn.style.flex = "1 1 0%";
            continuouslyBtn.innerText = "확인";
            clupPageBtn.style.display = "none";
            signalCheck = true;
        }
    });
});
