// 정렬 우선 순위 선택하는 버튼 클릭 시 모달 열기
const tabListBtn = document.querySelector(".tab-list-btn");
const tabList = document.querySelector(".tab-list");
const teenchinshow = document.querySelector(".teenchin-list")
let page = 1
const cansleButton = document.querySelector(".teenchin-more-btn")


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

// 틴친중, 신청중 버튼 클릭 시 발생 하는 이벤트
const teenchinModalWrap = document.querySelector(".teenchin-modal-wrap");
const teenchinModalContainer = document.querySelector(".teenchin-modal-container");

// 틴친중


// 신청중
const teenchinWaitBtns = document.querySelectorAll(".teenchin-wait-btn");

teenchinWaitBtns.forEach((teenchinWaitBtn) => {
    teenchinWaitBtn.addEventListener("click", (e) => {

    });
});

// 수락하기
const teenchinAgreeBtns = document.querySelectorAll(".teenchin-agree-btn");

teenchinAgreeBtns.forEach((teenchinAgreeBtn) => {
    teenchinAgreeBtn.addEventListener("click", (e) => {

    });
});

// 버튼의 클래스에 따라 모달창 내용을 변경해주는 함수
const teenchinModalTitle = document.querySelector(".teenchin-modal-title");
const teenchinModalCancleBtn = document.querySelector(".teenchin-modal-cancle-btn");
const teenchinModalChangedBtn = document.querySelector(".teenchin-modal-changed-btn");
const modalHeaderTitle = teenchinModalWrap.querySelector(".modal-header-title");

const madalContent = (e, profileName) => {
    if (e.target.closest(".teenchin-agree-btn")) {
        teenchinModalTitle.innerText = `${profileName}님의\n틴친 요청을 수락 하시겠습니까?`;
        teenchinModalCancleBtn.innerText = `거절하기`;
        teenchinModalCancleBtn.classList.add("refuse-btn");
        teenchinModalChangedBtn.innerText = `수락하기`;
        teenchinModalChangedBtn.classList.add("agree-btn");
        modalHeaderTitle.innerText = `${profileName}님의`;
        return;
    }

    if (e.target.closest(".teenchin-btn")) {
        teenchinModalTitle.innerText = `${profileName}님과\n틴친을 취소 하시겠습니까?`;
        teenchinModalChangedBtn.innerText = `틴친끊기`;
        modalHeaderTitle.innerText = `더 이상 ${profileName}님과\n틴친관계가 아닙니다.`;
        return;
    }

    if (e.target.closest(".teenchin-wait-btn")) {
        teenchinModalTitle.innerText = `${profileName}님에게\n보낸 틴친 신청을 취소하시겠습니까?`;
        teenchinModalChangedBtn.innerText = `취소하기`;
        modalHeaderTitle.innerText = `${profileName}님에게\n보낸 틴친 신청을 취소했습니다.`;
    }
};

// 틴친끊기, 취소하기 클릭 시 기존 모달을 닫고 확인모달을 나타내는 이벤트
const teenchinModalBtns = teenchinModalContainer.querySelectorAll("button");
const teenchinCheckModalContainer = teenchinModalWrap.querySelector(".teenchin-check-modal-container");

teenchinModalBtns.forEach((teenchinModalBtn) => {
    teenchinModalBtn.addEventListener("click", (e) => {
        teenchinModalContainer.style.animation = "popDown 0.5s";
        if (e.target.closest(".teenchin-modal-cancle-btn")) {
            if (e.target.closest(".teenchin-modal-cancle-btn").classList.contains("refuse-btn")) {
                modalHeaderTitle.innerText += `\n틴친 요청을 거절하셨습니다.`;

                setTimeout(() => {
                    teenchinModalCancleBtn.innerText = `닫기`;
                    teenchinModalCancleBtn.classList.remove("refuse-btn");
                    teenchinModalChangedBtn.classList.remove("agree-btn");
                    teenchinModalContainer.style.display = "none";
                    teenchinCheckModalContainer.style.animation = "popUp 0.5s";
                    teenchinCheckModalContainer.style.display = "flex";
                }, 500);
                return;
            }
            setTimeout(() => {
                teenchinModalWrap.style.display = "none";
            }, 500);
        } else {
            if (e.target.closest(".teenchin-modal-changed-btn").classList.contains("agree-btn")) {
                modalHeaderTitle.innerText += `\n틴친 요청을 수락하셨습니다.`;

                setTimeout(() => {
                    teenchinModalCancleBtn.innerText = `닫기`;
                    teenchinModalCancleBtn.classList.remove("refuse-btn");
                    teenchinModalChangedBtn.classList.remove("agree-btn");
                }, 500);
            }

            setTimeout(() => {
                teenchinModalContainer.style.display = "none";
                teenchinCheckModalContainer.style.animation = "popUp 0.5s";
                teenchinCheckModalContainer.style.display = "flex";
            }, 500);
        }
    });
});

// 2번째 확인 모달 내 확인버튼 클릭 시 발생하는 이벤트
const checkBtn = teenchinModalWrap.querySelector(".check-btn");

checkBtn.addEventListener("click", () => {
    teenchinCheckModalContainer.style.animation = "popDown 0.5s";
    setTimeout(() => {
        teenchinModalContainer.style.display = "flex";
        teenchinCheckModalContainer.style.display = "none";
        teenchinModalWrap.style.display = "none";
    }, 500);
});

// 쪽지 보내기 클릭 시 작성 모달 여는 이벤트
const sendModalWrap = document.querySelector(".send-modal-wrap");
const sendLetterBtns = document.querySelectorAll(".send-letter-btn");

sendLetterBtns.forEach((sendLetterBtn) => {
    sendLetterBtn.addEventListener("click", (e) => {

    });
});

// textarea 내 value에 따라 보내기 버튼 활성화/비활성화
const sendCheckBtn = document.querySelector(".send-check-btn");
const textarea = document.querySelector("textarea[name=send-content]");

textarea.addEventListener("input", (e) => {
    checkValue();
});

const checkValue = () => {
    if (textarea.value) {
        sendCheckBtn.disabled = false;
        return;
    }
    sendCheckBtn.disabled = true;
};

// 쪽지 보내기 모달 내 버튼 클릭 시 발생하는 이벤트
const sendModalBtns = document.querySelectorAll(".send-modal-container button");

sendModalBtns.forEach((sendModalBtn) => {
    sendModalBtn.addEventListener("click", (e) => {
        if (e.target.className == "send-check-btn") {
            sendModalWrap.querySelector(".send-modal-container").style.animation = "popDown 0.5s";
            setTimeout(() => {
                sendModalWrap.querySelector(".send-modal-container").style.display = "none";
                sendModalWrap.querySelector(".check-modal-container").style.animation = "popUp 0.5s";
                sendModalWrap.querySelector(".check-modal-container").style.display = "flex";
                sendModalWrap.querySelector(".send-receiver-email").value = ``;
                sendModalWrap.querySelector("textarea[name=send-content]").value = ``;
                checkValue();
            }, 450);
        } else {
            sendModalWrap.querySelector(".send-modal-container").style.animation = "popDown 0.5s";
            setTimeout(() => {
                sendModalWrap.style.display = "none";
                sendModalWrap.querySelector(".send-receiver-email").value = ``;
                sendModalWrap.querySelector("textarea[name=send-content]").value = ``;
                checkValue();
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










const showList = (teenchin) =>{
    let text = '';
    console.log(teenchin)
    if (teenchin.length ===0){
            text +=
            `<div class="signal-none">아직 새로운 틴친신청이 없습니다.</div>`
        }
    else{

        teenchin.forEach((teenchin)=>{
            if (teenchin.is_friend === 1 && teenchin.receiver_id === parseInt(member_id) && teenchin.sender__memberprofile__profile_path ===null ){
            text += `
            <div class="teenchin-box">
                <div class="teenchin-items">
                    <!-- 틴친 상세프로필 이동 주소 필요 -->
                    <div>
                        <div class="profile-img-contents">
                            <div class="profile-img-box">
                                <img class="profile-img" src="/static/public/web/images/logo/logo1.png" alt="" />
                            </div>
                            <div class="profile-img-gap"></div>
                        </div>
                        <div class="profile-info-contents">
                            <div class="profile-name">${teenchin.sender__member_nickname}</div>
                            <div class="profile-info-item">
                                <div style="margin-right: 10px">
                                    <span>활동</span>
                                    <span>${teenchin.aactivity_countss}</span>
                                </div>
                                <span>|</span>
                                <div style="margin-left: 10px">
                                    <span>모임</span>
                                    <span>${teenchin.club_countss}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="teenchin-btn-container">
                        <div class="teenchin-btn-box">
                            <button class="teenchin-btn ${teenchin.sender_id}" type="button">
                                <svg xmlns="http://www.w3.org/2000/svg" class="check-svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                </svg>
                                <span>틴친중</span>
                            </button>
                            <button class="send-letter-btn ${teenchin.sender_id}" type="button">
                                <svg aria-label="게시물 공유" class="send-svg" fill="currentColor" height="24" role="img" viewBox="0 0 24 24" width="24">
                                    <line fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="2" x1="22" x2="9.218" y1="3" y2="10.083"></line>
                                    <polygon fill="none" points="11.698 20.334 22 3.001 2 3.001 9.218 10.084 11.698 20.334" stroke="currentColor" stroke-linejoin="round" stroke-width="2"></polygon>
                                </svg>
                                <span>쪽지 보내기</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>`}
            else if (teenchin.is_friend === 1 && teenchin.receiver_id === parseInt(member_id)&& teenchin.sender__memberprofile__profile_path !== null ){
            text += `
            <div class="teenchin-box">
                <div class="teenchin-items">
                    <!-- 틴친 상세프로필 이동 주소 필요 -->
                    <div >
                        <div class="profile-img-contents">
                            <div class="profile-img-box">
                                <img class="profile-img" src="/upload/${teenchin.sender__memberprofile__profile_path}" alt="" />
                            </div>
                            <div class="profile-img-gap"></div>
                        </div>
                        <div class="profile-info-contents">
                            <div class="profile-name">${teenchin.sender__member_nickname}</div>
                            <div class="profile-info-item">
                                <div style="margin-right: 10px">
                                    <span>활동</span>
                                    <span>${teenchin.aactivity_countss}</span>
                                </div>
                                <span>|</span>
                                <div style="margin-left: 10px">
                                    <span>모임</span>
                                    <span>${teenchin.club_countss}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="teenchin-btn-container">
                        <div class="teenchin-btn-box">
                            <button class="teenchin-btn ${teenchin.sender_id}" type="button">
                                <svg xmlns="http://www.w3.org/2000/svg" class="check-svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                </svg>
                                <span>틴친중</span>
                            </button>
                            <button class="send-letter-btn ${teenchin.sender_id}" type="button">
                                <svg aria-label="게시물 공유" class="send-svg" fill="currentColor" height="24" role="img" viewBox="0 0 24 24" width="24">
                                    <line fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="2" x1="22" x2="9.218" y1="3" y2="10.083"></line>
                                    <polygon fill="none" points="11.698 20.334 22 3.001 2 3.001 9.218 10.084 11.698 20.334" stroke="currentColor" stroke-linejoin="round" stroke-width="2"></polygon>
                                </svg>
                                <span>쪽지 보내기</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>`}
            else if (teenchin.is_friend === 1 && teenchin.sender_id=== parseInt(member_id) && teenchin.receiver__memberprofile__profile_path ===null){
            text += `
            <div class="teenchin-box">
                <div class="teenchin-items">
                    <!-- 틴친 상세프로필 이동 주소 필요 -->
                    <div >
                        <div class="profile-img-contents">
                            <div class="profile-img-box">
                                <img class="profile-img" src="/static/public/web/images/logo/logo1.png" alt="" />
                            </div>
                            <div class="profile-img-gap"></div>
                        </div>
                        <div class="profile-info-contents">
                            <div class="profile-name">${teenchin.receiver__member_nickname}</div>
                            <div class="profile-info-item">
                                <div style="margin-right: 10px">
                                    <span>활동</span>
                                    <span>${teenchin.aactivity_count}</span>
                                </div>
                                <span>|</span>
                                <div style="margin-left: 10px">
                                    <span>모임</span>
                                    <span>${teenchin.club_count}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="teenchin-btn-container">
                        <div class="teenchin-btn-box">
                            <button class="teenchin-btn ${teenchin.receiver_id}" type="button">
                                <svg xmlns="http://www.w3.org/2000/svg" class="check-svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                </svg>
                                <span>틴친중</span>
                            </button>
                            <button class="send-letter-btn ${teenchin.receiver_id}" type="button">
                                <svg aria-label="게시물 공유" class="send-svg" fill="currentColor" height="24" role="img" viewBox="0 0 24 24" width="24">
                                    <line fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="2" x1="22" x2="9.218" y1="3" y2="10.083"></line>
                                    <polygon fill="none" points="11.698 20.334 22 3.001 2 3.001 9.218 10.084 11.698 20.334" stroke="currentColor" stroke-linejoin="round" stroke-width="2"></polygon>
                                </svg>
                                <span>쪽지 보내기</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>`}
            else if (teenchin.is_friend === 1 && teenchin.sender_id=== parseInt(member_id) && teenchin.receiver__memberprofile__profile_path !==null){
            text += `
            <div class="teenchin-box">
                <div class="teenchin-items">
                    <!-- 틴친 상세프로필 이동 주소 필요 -->
                    <div >
                        <div class="profile-img-contents">
                            <div class="profile-img-box">
                                <img class="profile-img" src="/upload/${teenchin.receiver__memberprofile__profile_path}" alt="" />
                            </div>
                            <div class="profile-img-gap"></div>
                        </div>
                        <div class="profile-info-contents">
                            <div class="profile-name">${teenchin.receiver__member_nickname}</div>
                            <div class="profile-info-item">
                                <div style="margin-right: 10px">
                                    <span>활동</span>
                                    <span>${teenchin.aactivity_count}</span>
                                </div>
                                <span>|</span>
                                <div style="margin-left: 10px">
                                    <span>모임</span>
                                    <span>${teenchin.club_count}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="teenchin-btn-container">
                        <div class="teenchin-btn-box">
                            <button class="teenchin-btn ${teenchin.receiver_id}" type="button">
                                <svg xmlns="http://www.w3.org/2000/svg" class="check-svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                </svg>
                                <span>틴친중</span>
                            </button>
                            <button class="send-letter-btn ${teenchin.receiver_id}" type="button">
                                <svg aria-label="게시물 공유" class="send-svg" fill="currentColor" height="24" role="img" viewBox="0 0 24 24" width="24">
                                    <line fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="2" x1="22" x2="9.218" y1="3" y2="10.083"></line>
                                    <polygon fill="none" points="11.698 20.334 22 3.001 2 3.001 9.218 10.084 11.698 20.334" stroke="currentColor" stroke-linejoin="round" stroke-width="2"></polygon>
                                </svg>
                                <span>쪽지 보내기</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>`}
             else if(teenchin.is_friend === -1 && teenchin.receiver_id === parseInt(member_id) && teenchin.sender__memberprofile__profile_path ===null){
                 text += `
                 <div class="teenchin-box">
                    <div class="teenchin-items">
                        <!-- 틴친 상세프로필 이동 주소 필요 -->
                        <div>
                            <div class="profile-img-contents">
                                <div class="profile-img-box">
                                    <img class="profile-img" src="/static/public/web/images/logo/logo1.png" alt="HR에듀센터_인사드림(insadream)" />
                                </div>
                                <div class="profile-img-gap"></div>
                            </div>
                            <div class="profile-info-contents">
                                <div class="profile-name">${teenchin.sender__member_nickname}</div>
                                <div class="profile-info-item">
                                    <div style="margin-right: 10px">
                                        <span>활동</span>
                                        <span>${teenchin.aactivity_countss}</span>
                                    </div>
                                    <span>|</span>
                                    <div style="margin-left: 10px">
                                        <span>모임</span>
                                        <span>${teenchin.club_countss}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="teenchin-btn-container">
                            <div class="teenchin-btn-box">
                                <button class="teenchin-agree-btn ${teenchin.sender_id}" type="button">
                                    <svg aria-label="사람 찾아보기" class="teenchin-agree-svg" fill="currentColor" height="24" role="img" viewBox="0 0 48 48" width="24">
                                        <path
                                            d="M32 25.5c5.2 0 9.5-4.3 9.5-9.5S37.2 6.5 32 6.5s-9.5 4.3-9.5 9.5 4.3 9.5 9.5 9.5zm0-16c3.6 0 6.5 2.9 6.5 6.5s-2.9 6.5-6.5 6.5-6.5-2.9-6.5-6.5 2.9-6.5 6.5-6.5zm5.5 19h-11c-5.5 0-10 4.5-10 10V40c0 .8.7 1.5 1.5 1.5s1.5-.7 1.5-1.5v-1.5c0-3.9 3.1-7 7-7h11c3.9 0 7 3.1 7 7V40c0 .8.7 1.5 1.5 1.5s1.5-.7 1.5-1.5v-1.5c0-5.5-4.5-10-10-10zm-20-4.5c0-.8-.7-1.5-1.5-1.5h-5.5V17c0-.8-.7-1.5-1.5-1.5s-1.5.7-1.5 1.5v5.5H2c-.8 0-1.5.7-1.5 1.5s.7 1.5 1.5 1.5h5.5V31c0 .8.7 1.5 1.5 1.5s1.5-.7 1.5-1.5v-5.5H16c.8 0 1.5-.7 1.5-1.5z"
                                        ></path>
                                    </svg>
                                    <span>수락대기</span>
                                </button>
                                <button class="send-letter-btn ${teenchin.sender_id}" type="button">
                                    <svg aria-label="게시물 공유" class="send-svg" fill="currentColor" height="24" role="img" viewBox="0 0 24 24" width="24">
                                        <line fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="2" x1="22" x2="9.218" y1="3" y2="10.083"></line>
                                        <polygon fill="none" points="11.698 20.334 22 3.001 2 3.001 9.218 10.084 11.698 20.334" stroke="currentColor" stroke-linejoin="round" stroke-width="2"></polygon>
                                    </svg>
                                    <span>쪽지 보내기</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>`}
             else if(teenchin.is_friend === -1 && teenchin.receiver_id === parseInt(member_id) && teenchin.sender__memberprofile__profile_path !==null){
                 text += `
                 <div class="teenchin-box">
                    <div class="teenchin-items">
                        <!-- 틴친 상세프로필 이동 주소 필요 -->
                        <div>
                            <div class="profile-img-contents">
                                <div class="profile-img-box">
                                    <img class="profile-img" src="/upload/${teenchin.sender__memberprofile__profile_path}" alt="HR에듀센터_인사드림(insadream)" />
                                </div>
                                <div class="profile-img-gap"></div>
                            </div>
                            <div class="profile-info-contents">
                                <div class="profile-name">${teenchin.sender__member_nickname}</div>
                                <div class="profile-info-item">
                                    <div style="margin-right: 10px">
                                        <span>활동</span>
                                        <span>${teenchin.aactivity_countss}</span>
                                    </div>
                                    <span>|</span>
                                    <div style="margin-left: 10px">
                                        <span>모임</span>
                                        <span>${teenchin.club_countss}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="teenchin-btn-container">
                            <div class="teenchin-btn-box">
                                <button class="teenchin-agree-btn ${teenchin.sender_id}" type="button">
                                    <svg aria-label="사람 찾아보기" class="teenchin-agree-svg" fill="currentColor" height="24" role="img" viewBox="0 0 48 48" width="24">
                                        <path
                                            d="M32 25.5c5.2 0 9.5-4.3 9.5-9.5S37.2 6.5 32 6.5s-9.5 4.3-9.5 9.5 4.3 9.5 9.5 9.5zm0-16c3.6 0 6.5 2.9 6.5 6.5s-2.9 6.5-6.5 6.5-6.5-2.9-6.5-6.5 2.9-6.5 6.5-6.5zm5.5 19h-11c-5.5 0-10 4.5-10 10V40c0 .8.7 1.5 1.5 1.5s1.5-.7 1.5-1.5v-1.5c0-3.9 3.1-7 7-7h11c3.9 0 7 3.1 7 7V40c0 .8.7 1.5 1.5 1.5s1.5-.7 1.5-1.5v-1.5c0-5.5-4.5-10-10-10zm-20-4.5c0-.8-.7-1.5-1.5-1.5h-5.5V17c0-.8-.7-1.5-1.5-1.5s-1.5.7-1.5 1.5v5.5H2c-.8 0-1.5.7-1.5 1.5s.7 1.5 1.5 1.5h5.5V31c0 .8.7 1.5 1.5 1.5s1.5-.7 1.5-1.5v-5.5H16c.8 0 1.5-.7 1.5-1.5z"
                                        ></path>
                                    </svg>
                                    <span>수락대기</span>
                                </button>
                                <button class="send-letter-btn ${teenchin.sender_id}" type="button">
                                    <svg aria-label="게시물 공유" class="send-svg" fill="currentColor" height="24" role="img" viewBox="0 0 24 24" width="24">
                                        <line fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="2" x1="22" x2="9.218" y1="3" y2="10.083"></line>
                                        <polygon fill="none" points="11.698 20.334 22 3.001 2 3.001 9.218 10.084 11.698 20.334" stroke="currentColor" stroke-linejoin="round" stroke-width="2"></polygon>
                                    </svg>
                                    <span>쪽지 보내기</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>`}
             else if(teenchin.is_friend === -1 && teenchin.sender_id=== parseInt(member_id) && teenchin.receiver__memberprofile__profile_path ===null){
                 text += `
                 <div class="teenchin-box">
                    <div class="teenchin-items">
                        <!-- 틴친 상세프로필 이동 주소 필요 -->
                        <div>
                            <div class="profile-img-contents">
                                <div class="profile-img-box">
                                    <img class="profile-img" src="/static/public/web/images/logo/logo1.png" alt="HR에듀센터_인사드림(insadream)" />
                                </div>
                                <div class="profile-img-gap"></div>
                            </div>
                            <div class="profile-info-contents">
                                <div class="profile-name">${teenchin.receiver__member_nickname}</div>
                                <div class="profile-info-item">
                                    <div style="margin-right: 10px">
                                        <span>활동</span>
                                        <span>${teenchin.aactivity_count}</span>
                                    </div>
                                    <span>|</span>
                                    <div style="margin-left: 10px">
                                        <span>모임</span>
                                        <span>${teenchin.club_count}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="teenchin-btn-container">
                            <div class="teenchin-btn-box">
                                <button class="teenchin-wait-btn ${teenchin.receiver_id}" type="button">
                                    <svg class="request-svg" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                        <path
                                            d="M12 2C17.523 2 22 6.478 22 12C22 17.522 17.523 22 12 22C6.477 22 2 17.522 2 12C2 6.478 6.477 2 12 2ZM12 3.667C7.405 3.667 3.667 7.405 3.667 12C3.667 16.595 7.405 20.333 12 20.333C16.595 20.333 20.333 16.595 20.333 12C20.333 7.405 16.595 3.667 12 3.667ZM11.9987 14.5022C12.5502 14.5022 12.9973 14.9494 12.9973 15.5009C12.9973 16.0524 12.5502 16.4996 11.9987 16.4996C11.4471 16.4996 11 16.0524 11 15.5009C11 14.9494 11.4471 14.5022 11.9987 14.5022ZM11.9945 7C12.3742 6.9997 12.6882 7.2816 12.7381 7.64764L12.7451 7.7494L12.7487 12.251C12.749 12.6652 12.4135 13.0013 11.9993 13.0016C11.6196 13.0019 11.3055 12.72 11.2556 12.354L11.2487 12.2522L11.2451 7.7506C11.2447 7.33639 11.5802 7.00033 11.9945 7Z"
                                        ></path>
                                    </svg>
                                    <span>신청중</span>
                                </button>
                                <button class="send-letter-btn ${teenchin.receiver_id}" type="button">
                                    <svg aria-label="게시물 공유" class="send-svg" fill="currentColor" height="24" role="img" viewBox="0 0 24 24" width="24">
                                        <line fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="2" x1="22" x2="9.218" y1="3" y2="10.083"></line>
                                        <polygon fill="none" points="11.698 20.334 22 3.001 2 3.001 9.218 10.084 11.698 20.334" stroke="currentColor" stroke-linejoin="round" stroke-width="2"></polygon>
                                    </svg>
                                    <span>쪽지 보내기</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>`}
             else if(teenchin.is_friend === -1 && teenchin.sender_id=== parseInt(member_id) && teenchin.receiver__memberprofile__profile_path !==null){
                 text += `
                 <div class="teenchin-box">
                    <div class="teenchin-items">
                        <!-- 틴친 상세프로필 이동 주소 필요 -->
                        <div>
                            <div class="profile-img-contents">
                                <div class="profile-img-box">
                                    <img class="profile-img" src="/upload/${teenchin.receiver__memberprofile__profile_path}" alt="HR에듀센터_인사드림(insadream)" />
                                </div>
                                <div class="profile-img-gap"></div>
                            </div>
                            <div class="profile-info-contents">
                                <div class="profile-name">${teenchin.receiver__member_nickname}</div>
                                <div class="profile-info-item">
                                    <div style="margin-right: 10px">
                                        <span>활동</span>
                                        <span>${teenchin.aactivity_count}</span>
                                    </div>
                                    <span>|</span>
                                    <div style="margin-left: 10px">
                                        <span>모임</span>
                                        <span>${teenchin.club_count}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="teenchin-btn-container">
                            <div class="teenchin-btn-box">
                                <button class="teenchin-wait-btn ${teenchin.receiver_id}" type="button">
                                    <svg class="request-svg" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                        <path
                                            d="M12 2C17.523 2 22 6.478 22 12C22 17.522 17.523 22 12 22C6.477 22 2 17.522 2 12C2 6.478 6.477 2 12 2ZM12 3.667C7.405 3.667 3.667 7.405 3.667 12C3.667 16.595 7.405 20.333 12 20.333C16.595 20.333 20.333 16.595 20.333 12C20.333 7.405 16.595 3.667 12 3.667ZM11.9987 14.5022C12.5502 14.5022 12.9973 14.9494 12.9973 15.5009C12.9973 16.0524 12.5502 16.4996 11.9987 16.4996C11.4471 16.4996 11 16.0524 11 15.5009C11 14.9494 11.4471 14.5022 11.9987 14.5022ZM11.9945 7C12.3742 6.9997 12.6882 7.2816 12.7381 7.64764L12.7451 7.7494L12.7487 12.251C12.749 12.6652 12.4135 13.0013 11.9993 13.0016C11.6196 13.0019 11.3055 12.72 11.2556 12.354L11.2487 12.2522L11.2451 7.7506C11.2447 7.33639 11.5802 7.00033 11.9945 7Z"
                                        ></path>
                                    </svg>
                                    <span>신청중</span>
                                </button>
                                <button class="send-letter-btn ${teenchin.receiver_id}" type="button">
                                    <svg aria-label="게시물 공유" class="send-svg" fill="currentColor" height="24" role="img" viewBox="0 0 24 24" width="24">
                                        <line fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="2" x1="22" x2="9.218" y1="3" y2="10.083"></line>
                                        <polygon fill="none" points="11.698 20.334 22 3.001 2 3.001 9.218 10.084 11.698 20.334" stroke="currentColor" stroke-linejoin="round" stroke-width="2"></polygon>
                                    </svg>
                                    <span>쪽지 보내기</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>`}
        })
    }return text;
}





teenchinshow.addEventListener("click", async (e)=>{
    if (e.target.classList[0] === 'teenchin-btn'){
        const delet = e.target.classList[1]
        let profileName = e.target.closest(".teenchin-items").querySelector(".profile-name").innerText;
        madalContent(e, profileName);
        teenchinModalContainer.style.animation = "popUp 0.5s";
        teenchinModalWrap.style.display = "flex";
        teenchinModalChangedBtn.addEventListener("click", async(e)=>{
            teenchinService.remove(delet)
            await teenchinService.getList(member_id, 1,status_teenchin,search, showList).then((text) => {
                teenchinshow.innerHTML = text;
            });
            page = 1
        } )


    }if(e.target.classList[0] === 'teenchin-agree-btn'){
        const update = e.target.classList[1]
        let profileName = e.target.closest(".teenchin-items").querySelector(".profile-name").innerText;
        madalContent(e, profileName);
        teenchinModalContainer.style.animation = "popUp 0.5s";
        teenchinModalWrap.style.display = "flex";
        teenchinModalCancleBtn.addEventListener("click", async(e) =>{
            teenchinService.remove(update)
            await teenchinService.getList(member_id, 1,status_teenchin,search, showList).then((text) => {
                teenchinshow.innerHTML = text;
            });
            page = 1
            await teenchinService.getList(member_id, page,).then((teenchin) => {
            if (teenchin.length === 0){
            cansleButton.style.display = "none";}})
        })
         teenchinModalChangedBtn.addEventListener("click", async(e) =>{
            teenchinService.update(update)
            await teenchinService.getList(member_id, 1,status_teenchin,search, showList).then((text) => {
                teenchinshow.innerHTML = text;
            });
            page = 1
        })
    }if(e.target.classList[0] === 'teenchin-wait-btn'){
        const update = e.target.classList[1]
        let profileName = e.target.closest(".teenchin-items").querySelector(".profile-name").innerText;
        madalContent(e, profileName);
        teenchinModalContainer.style.animation = "popUp 0.5s";
        teenchinModalWrap.style.display = "flex";
        teenchinModalChangedBtn.addEventListener("click", async(e) =>{
            teenchinService.remove(update)
            await teenchinService.getList(member_id, 1,status_teenchin,search, showList).then((text) => {
                teenchinshow.innerHTML = text;
            });
            page = 1
        })
    }if(e.target.classList[0] === 'send-letter-btn'){
        const receiver = e.target.classList[1]
        document.querySelector(".send-receiver-email").value = e.target.closest(".teenchin-items").querySelector(".profile-name").innerText;
        sendModalWrap.querySelector(".send-modal-container").style.animation = "popUp 0.5s";
        sendModalWrap.style.display = "block";
        sendCheckBtn.addEventListener("click", async(e)=>{
            const sendText = document.getElementById("send-content")
            await teenchinService.write({
                letter_content: sendText.value,
                receiver_id : receiver
            })
            await teenchinService.getList(member_id, 1,status_teenchin,search, showList).then((text) => {
                teenchinshow.innerHTML = text;
            });
            page = 1

        })
    }

})


// 페이지 로드 시 실행되는 부분





cansleButton.addEventListener("click", (e) => {
    teenchinService.getList(member_id, ++page,status_teenchin,search, showList).then((text) => {
        teenchinshow.innerHTML += text;
    });

    teenchinService.getList(member_id, page + 1,).then((teenchin) => {
    if (teenchin.length === 0){
        cansleButton.style.display = "none";
    }

});

});





teenchinService.getList(member_id, page,status_teenchin,search, showList).then((text) => {
    teenchinshow.innerHTML += text;
});


teenchinService.getList(member_id, page,status_teenchin,search,showList);


teenchinService.getList(member_id, page,status_teenchin,search ).then((teenchin) => {
    console.log(teenchin.length)
if (teenchin.length === 0){
    cansleButton.style.display = "none";
}})
teenchinService.getList(member_id, page+1,status_teenchin,search).then((teenchin) => {
    console.log(teenchin.length)
if (teenchin.length === 0){
    cansleButton.style.display = "none";
}})