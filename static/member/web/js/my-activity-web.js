// 활동목록, 관심목록 버튼 이벤트
const activityList = document.querySelector("#activity-list");
const interestList = document.querySelector("#interest-activity");
const activeBtn = document.querySelectorAll(".categori-btn");
const categoris = document.querySelectorAll(".activity-categories");
let page = 1
const inner = document.querySelector(".activity-list-wrap")


activeBtn.forEach((btn, i) => {
    btn.addEventListener("click", () => {
        if (i == 0) {
            // 활동목록 클릭시
            categoris[0].classList.add("activity-categori-on");
            categoris[1].classList.remove("activity-categori-on");

        } else {
            // 관심목록 클릭시
            categoris[1].classList.add("activity-categori-on");
            categoris[0].classList.remove("activity-categori-on");

        }
    });
});

// 활동 목록 관심버튼(하트) 클릭 이벤트
const likeBtn = document.querySelectorAll(".like-btn");
const likeOn = document.querySelectorAll(".like-on");
const likeOff = document.querySelectorAll(".like-off");
const likeModal = document.querySelector(".like-modal-wrap");
const likeCloseBtn = document.querySelector(".modal-like-button");

// console.log(likeBtn);
// console.log(likeOn);
// console.log(likeOff);

likeBtn.forEach((btn, i) => {
    // 관심버튼(하트)을 눌렀을때
    btn.addEventListener("click", () => {
        // likeBtn의 갯수만큼 for문으로 반복해서
        for (let j = 0; j < likeBtn.length; j++) {
            // 만약 누른 버튼의 인덱스번호가 관심버튼의 인덱스 번호와 같다면
            if (i == j) {
                if (!likeOff[j].classList.contains("like-none")) {
                    // 만약 관심버튼이 none상태(좋아요가 눌리지 않은 상태)라면
                    likeOff[j].classList.add("like-none");
                    likeOn[j].classList.remove("like-none");
                    // 관심목록 추가 모달
                    likeModalOn();
                } else {
                    // 관심버튼이 눌러진 상태라면
                    likeOff[j].classList.remove("like-none");
                    likeOn[j].classList.add("like-none");
                    // 관심목록 해제 모달
                    likeModalOff();
                }
            }
        }
    });
});

// 관심활동에서 하트 눌렀을때
const likeButton = document.querySelectorAll(".like-button");
const interOn = document.querySelectorAll(".like-on-interest");
const interOff = document.querySelectorAll(".like-off-interest");
const activityDivs = document.querySelectorAll(".activity-wrap-interest");
const unlikeModal = document.querySelector(".unlike-modal-wrap");
const unlikeCloseBtn = document.querySelector(".modal-unlike-button");
const myMain = document.querySelector(".mypage-main");

likeButton.forEach((btn, i) => {
    // 관심버튼(하트)을 눌렀을때
    btn.addEventListener("click", () => {
        // likeBtn의 갯수만큼 for문으로 반복해서
        for (let j = 0; j < likeBtn.length; j++) {
            // 만약 누른 버튼의 인덱스번호가 관심버튼의 인덱스 번호와 같다면
            if (i == j) {
                if (!interOff[j].classList.contains("like-none")) {
                    interOff[j].classList.add("like-none");
                    interOn[j].classList.remove("like-none");
                } else {
                    interOff[j].classList.remove("like-none");
                    interOn[j].classList.add("like-none");
                    // 좋아요 해제 시 모달 (삭제되어야 할 div를 매개변수로 전달)
                    unlikeModalOn(activityDivs[j]);
                }
            }
        }
    });
});

// 활동 목록에서 관심버튼 눌렀을때 함수
function likeModalOn() {
    likeModal.style.display = "flex";
    likeModal.style.backgroundColor = "rgba(0,0,0,.5)";
    // 닫기 버튼 눌렀을 때
    likeCloseBtn.addEventListener("click", () => {
        myMain.style.opacity = "1";
        likeModal.style.display = "none";
    });
}
// 활동 목록에서 관심버튼 해제시 함수
function likeModalOff() {
    unlikeModal.style.display = "flex";
    unlikeModal.style.backgroundColor = "rgba(0,0,0,.5)";
    // 닫기 버튼 눌렀을 때
    unlikeCloseBtn.addEventListener("click", () => {
        myMain.style.opacity = "1";
        unlikeModal.style.display = "none";
    });
}

// 관심 목록에서 관심버튼 눌러서 좋아요 해제시
function unlikeModalOn(activityDiv) {
    unlikeModal.style.display = "flex";
    unlikeModal.style.backgroundColor = "rgba(0,0,0,.5)";
    // 닫기 버튼 눌렀을 때
    unlikeCloseBtn.addEventListener("click", () => {
        myMain.style.opacity = "1";
        unlikeModal.style.display = "none";
        // 매개변수로 받아온 div를 삭제
        activityDiv.remove();
    });
}




const showList = async (activity_data) =>{
    let text='';
    if (activity_data.length ===0){
        text += `<div class="signal-none">아직 새로운 활동이 없습니다.</div>`
    }
    else {
    activity_data.forEach((activity_data, i)=>{
        console.log(activity_data)
        const activityEndDate = new Date(activity_data.activity_end);
        const currentDate = new Date();
        // 시간 정보를 무시하고 일자만을 비교
        activityEndDate.setHours(0, 0, 0, 0);
        currentDate.setHours(0, 0, 0, 0);
        // activityEndDate >= currentDate 이게 아직 안한거임
        // 끝난거
        if (activityEndDate < currentDate && activity_data.thumbnail_path === ""){
        text += `<div class="activity-wrap">
                    <div class="activity-box">
                      <div class="activity-img-box">
                        <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="activity-img-link">
                          <!-- 종료된 활동은 이미지와 제목이 회색이라 클래스명이 다름 -->
                          <!-- <img class="passivity-img" /> -->
                          <img
                            src="/static/public/web/images/logo/logo8.png"
                            alt="ces올인원 패키지"
                            class="passivity-img"
                          />
                        </a>
                        <div class="like-btn-container">
                          <button type="button" class="like-btn " >
                            <span class="like-on like-none">
                              <svg xmlns="http://www.w3.org/2000/svg" class="like-img-container ${activity_data.id} ${ activity_data.status }" viewBox="0 0 20 20" fill="currentColor">
                                <path
                                  fill-rule="evenodd"
                                  d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z"
                                  clip-rule="evenodd"
                                  class="like-img-container ${activity_data.id} ${ activity_data.status }"
                                ></path>
                              </svg>
                            </span>
                            <span class="like-off">
                            <input type="hidden" name="is-like" value="">
                              <svg
                                value = ${activity_data.status}
                                id="like-target"
                                xmlns="http://www.w3.org/2000/svg"
                                class="unlike-img-container ${activity_data.id} ${ activity_data.status }"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                              >
                                <path
                                  stroke-linecap="round"
                                  stroke-linejoin="round"
                                  stroke-width="2"
                                  d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                                ></path>
                              </svg>
                            </span>
                          </button>
                        </div>
                      </div>
                      <div class="content-txt-box">
                        <div class="middle-txt">
                          <div>
                            <span class="txt-head-end">종료</span>
                          </div>
                          <div>
                            <span class="txt-head">${changeDate(activity_data.activity_end)}</span>
                          </div>
                          <div class="middle-txt-online">
                          </div>
                        </div>
                        <div class="content-tit-wrap">
                          <!-- 이 부분도 회색 -->
                          <!-- <a class="passivity-content-tit" -->
                          <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="passivity-content-tit">${activity_data.activity_title}</a>
                        </div>
                        <div class="bottom-content-wrap">
                          <div class="bottom-content-ahead">
                            <span></span>
                          </div>
                          <div class="bottom-content-back">
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>`
            }
        else if (activityEndDate < currentDate && activity_data.thumbnail_path !== ""){
        text += `<div class="activity-wrap">
                    <div class="activity-box">
                      <div class="activity-img-box">
                        <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="activity-img-link">
                          <!-- 종료된 활동은 이미지와 제목이 회색이라 클래스명이 다름 -->
                          <!-- <img class="passivity-img" /> -->
                          <img
                            src="/upload/${activity_data.thumbnail_path}"
                            alt="ces올인원 패키지"
                            class="passivity-img"
                          />
                        </a>
                        <div class="like-btn-container">
                          <button type="button" class="like-btn " >
                            <span class="like-on like-none">
                              <svg xmlns="http://www.w3.org/2000/svg" class="like-img-container ${activity_data.id} ${ activity_data.status }" viewBox="0 0 20 20" fill="currentColor">
                                <path
                                  fill-rule="evenodd"
                                  d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z"
                                  clip-rule="evenodd"
                                  class="like-img-container ${activity_data.id} ${ activity_data.status }"
                                ></path>
                              </svg>
                            </span>
                            <span class="like-off">
                            <input type="hidden" name="is-like" value="">
                              <svg
                                value = ${activity_data.status}
                                id="like-target"
                                xmlns="http://www.w3.org/2000/svg"
                                class="unlike-img-container ${activity_data.id} ${ activity_data.status }"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                              >
                                <path
                                  stroke-linecap="round"
                                  stroke-linejoin="round"
                                  stroke-width="2"
                                  d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                                ></path>
                              </svg>
                            </span>
                          </button>
                        </div>
                      </div>
                      <div class="content-txt-box">
                        <div class="middle-txt">
                          <div>
                            <span class="txt-head-end">종료</span>
                          </div>
                          <div>
                            <span class="txt-head">${changeDate(activity_data.activity_end)}</span>
                          </div>
                          <div class="middle-txt-online">
                          </div>
                        </div>
                        <div class="content-tit-wrap">
                          <!-- 이 부분도 회색 -->
                          <!-- <a class="passivity-content-tit" -->
                          <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="passivity-content-tit">${activity_data.activity_title}</a>
                        </div>
                        <div class="bottom-content-wrap">
                          <div class="bottom-content-ahead">
                            <span></span>
                          </div>
                          <div class="bottom-content-back">
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>`
            }
        // 참여자 + status(1) 참여확정
        else if (activityEndDate >= currentDate && activity_data.activitymember__status ===1 && activity_data.thumbnail_path ===''){
            text +=
                `<div class="activity-wrap">
                    <div class="activity-box">
                      <div class="activity-img-box">
                        <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="activity-img-link">
                          <img
                            src="/static/public/web/images/logo/logo8.png"
                            alt="ces올인원 패키지"
                            class="activity-img"
                          />
                        </a>
                        <div class="like-btn-container">
                          <button type="button" class="like-btn ${activity_data.id}">
                            <span class="like-on like-none">
                              <svg xmlns="http://www.w3.org/2000/svg" class="like-img-container ${activity_data.id} ${ activity_data.status }" viewBox="0 0 20 20" fill="currentColor">
                                <path
                                  class="like-img-container ${activity_data.id} ${ activity_data.status }"
                                  fill-rule="evenodd"
                                  d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z"
                                  clip-rule="evenodd"
                                ></path>
                              </svg>
                            </span>
                            <span class="like-off">
                            <input type="hidden" name="is-like" value="${ activity_data.status }">
                              <svg
                                value = ${activity_data.status}
                                name = is-like
                                 id="like-target"
                                xmlns="http://www.w3.org/2000/svg"
                                class="unlike-img-container ${activity_data.id} ${ activity_data.status }"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                              >
                                <path
                                  stroke-linecap="round"
                                  stroke-linejoin="round"
                                  stroke-width="2"
                                  d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                                ></path>
                              </svg>
                            </span>
                          </button>
                        </div>
                      </div>
                      <div class="content-txt-box">
                        <div class="middle-txt">
                          <div>
                            <span class="txt-head-confirm">참가확정</span>
                          </div>
                          <div>
                            <span class="txt-head">${changeDate(activity_data.activity_end)}</span>
                          </div>
                          <div class="middle-txt-online">
                          </div>
                        </div>
                        <div class="content-tit-wrap">
                          <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="content-tit">${activity_data.activity_title}</a>
                        </div>
                        <div class="bottom-content-wrap">
                          <div class="bottom-content-ahead">
                            <span></span>
                          </div>
                          <div class="bottom-content-back">
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>`
        }
        else if (activityEndDate >= currentDate && activity_data.activitymember__status ===1 && activity_data.thumbnail_path!==''){
            text +=
                `<div class="activity-wrap">
                    <div class="activity-box">
                      <div class="activity-img-box">
                        <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="activity-img-link">
                          <img
                            src="/upload/${activity_data.thumbnail_path}"
                            alt="ces올인원 패키지"
                            class="activity-img"
                          />
                        </a>
                        <div class="like-btn-container">
                          <button type="button" class="like-btn ${activity_data.id}">
                            <span class="like-on like-none">
                              <svg xmlns="http://www.w3.org/2000/svg" class="like-img-container ${activity_data.id} ${ activity_data.status }" viewBox="0 0 20 20" fill="currentColor">
                                <path
                                  class="like-img-container ${activity_data.id} ${ activity_data.status }"
                                  fill-rule="evenodd"
                                  d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z"
                                  clip-rule="evenodd"
                                ></path>
                              </svg>
                            </span>
                            <span class="like-off">
                            <input type="hidden" name="is-like" value="${ activity_data.status }">
                              <svg
                                value = ${activity_data.status}
                                name = is-like
                                 id="like-target"
                                xmlns="http://www.w3.org/2000/svg"
                                class="unlike-img-container ${activity_data.id} ${ activity_data.status }"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                              >
                                <path
                                  stroke-linecap="round"
                                  stroke-linejoin="round"
                                  stroke-width="2"
                                  d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                                ></path>
                              </svg>
                            </span>
                          </button>
                        </div>
                      </div>
                      <div class="content-txt-box">
                        <div class="middle-txt">
                          <div>
                            <span class="txt-head-confirm">참가확정</span>
                          </div>
                          <div>
                            <span class="txt-head">${changeDate(activity_data.activity_end)}</span>
                          </div>
                          <div class="middle-txt-online">
                          </div>
                        </div>
                        <div class="content-tit-wrap">
                          <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="content-tit">${activity_data.activity_title}</a>
                        </div>
                        <div class="bottom-content-wrap">
                          <div class="bottom-content-ahead">
                            <span></span>
                          </div>
                          <div class="bottom-content-back">
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>`
        }
        // // 참여자 + status(-1) 참여대기
        else if (activityEndDate >= currentDate && activity_data.activitymember__status ===-1 && activity_data.thumbnail_path===''){
            text += `
            <div class="activity-wrap">
                <div class="activity-box">
                  <div class="activity-img-box">
                    <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="activity-img-link">
                      <img
                        src="/static/public/web/images/logo/logo8.png"
                        alt="ces올인원 패키지"
                        class="activity-img"
                      />
                    </a>
                    <div class="like-btn-container">
                      <button type="button" class="like-btn ${activity_data.id}">
                        <span class="like-on like-none">
                          <svg xmlns="http://www.w3.org/2000/svg" class="like-img-container ${activity_data.id} ${ activity_data.status }" viewBox="0 0 20 20" fill="currentColor">
                            <path
                              class="like-img-container ${activity_data.id} ${ activity_data.status }"
                              fill-rule="evenodd"
                              d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z"
                              clip-rule="evenodd"
                            ></path>
                          </svg>
                        </span>
                        <span class="like-off">
                        <input type="hidden" name="is-like" value="${ activity_data.status }">
                          <svg
                            value = ${activity_data.status}
                            name = is-like
                             id="like-target"
                            xmlns="http://www.w3.org/2000/svg"
                            class="unlike-img-container ${activity_data.id} ${ activity_data.status }"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              stroke-linecap="round"
                              stroke-linejoin="round"
                              stroke-width="2"
                              d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                            ></path>
                          </svg>
                        </span>
                      </button>
                    </div>
                  </div>
                  <div class="content-txt-box">
                    <div class="middle-txt">
                      <div>
                        <span class="txt-head-wait">참가대기</span>
                      </div>
                      <div>
                        <span class="txt-head">${changeDate(activity_data.activity_end)}</span>
                      </div>
                      <div class="middle-txt-online">
                      </div>
                    </div>
                    <div class="content-tit-wrap">
                      <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="content-tit">${activity_data.activity_title}</a>
                    </div>
                    <div class="bottom-content-wrap">
                      <div class="bottom-content-ahead">
                        <span></span>
                      </div>
                      <div class="bottom-content-back">
                      </div>
                    </div>
                  </div>
                </div>
              </div>`
        }
        else if (activityEndDate >= currentDate && activity_data.activitymember__status ===-1 && activity_data.thumbnail_path !==''){
            text += `
            <div class="activity-wrap">
                <div class="activity-box">
                  <div class="activity-img-box">
                    <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="activity-img-link">
                      <img
                        src="/upload/${activity_data.thumbnail_path}"
                        alt="ces올인원 패키지"
                        class="activity-img"
                      />
                    </a>
                    <div class="like-btn-container">
                      <button type="button" class="like-btn ${activity_data.id}">
                        <span class="like-on like-none">
                          <svg xmlns="http://www.w3.org/2000/svg" class="like-img-container ${activity_data.id} ${ activity_data.status }" viewBox="0 0 20 20" fill="currentColor">
                            <path
                              class="like-img-container ${activity_data.id} ${ activity_data.status }"
                              fill-rule="evenodd"
                              d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z"
                              clip-rule="evenodd"
                            ></path>
                          </svg>
                        </span>
                        <span class="like-off">
                        <input type="hidden" name="is-like" value="${ activity_data.status }">
                          <svg
                            value = ${activity_data.status}
                            name = is-like
                             id="like-target"
                            xmlns="http://www.w3.org/2000/svg"
                            class="unlike-img-container ${activity_data.id} ${ activity_data.status }"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              stroke-linecap="round"
                              stroke-linejoin="round"
                              stroke-width="2"
                              d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                            ></path>
                          </svg>
                        </span>
                      </button>
                    </div>
                  </div>
                  <div class="content-txt-box">
                    <div class="middle-txt">
                      <div>
                        <span class="txt-head-wait">참가대기</span>
                      </div>
                      <div>
                        <span class="txt-head">${changeDate(activity_data.activity_end)}</span>
                      </div>
                      <div class="middle-txt-online">
                      </div>
                    </div>
                    <div class="content-tit-wrap">
                      <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="content-tit">${activity_data.activity_title}</a>
                    </div>
                    <div class="bottom-content-wrap">
                      <div class="bottom-content-ahead">
                        <span></span>
                      </div>
                      <div class="bottom-content-back">
                      </div>
                    </div>
                  </div>
                </div>
              </div>`
        //     개최좌
        }
        else if (activityEndDate >= currentDate && activity_data.club__member_id===parseInt(member_id) && activity_data.thumbnail_path === ''){
            text += `
            <div class="activity-wrap">
                <div class="activity-box">
                  <div class="activity-img-box">
                    <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="activity-img-link">
                      <img
                        src="/static/public/web/images/logo/logo8.png"
                        alt="ces올인원 패키지"
                        class="activity-img"
                      />
                    </a>
                    <div class="like-btn-container">
                      <button type="button" class="like-btn ${activity_data.id}" >
                        <span class="like-on like-none">
                          <svg xmlns="http://www.w3.org/2000/svg" class="like-img-container ${activity_data.id} ${ activity_data.status }" viewBox="0 0 20 20" fill="currentColor">
                            <path
                              fill-rule="evenodd"
                              d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z"
                              clip-rule="evenodd"
                              class="like-img-container ${activity_data.id} ${ activity_data.status }"
                            ></path>
                          </svg>
                        </span>
                        <span class="like-off">
                        <input type="hidden" name="is-like" value="${ activity_data.status } ">
                          <svg
                            id="like-target"
                            xmlns="http://www.w3.org/2000/svg"
                            class="unlike-img-container ${activity_data.id} ${ activity_data.status }"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              stroke-linecap="round"
                              stroke-linejoin="round"
                              stroke-width="2"
                              d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                            ></path>
                          </svg>
                        </span>
                      </button>
                    </div>
                  </div>
                  <div class="content-txt-box">
                    <div class="middle-txt">
                      <div>
                        <a class="txt-head-confirm" href="http://127.0.0.1:10000/member/activity/?activity_id=${activity_data.id}" >활동수정</a>
                      </div>
                      <div>
                        <span class="txt-head">${changeDate(activity_data.activity_end)}</span>
                      </div>
                      <div class="middle-txt-online">
                      </div>
                    </div>
                    <div class="content-tit-wrap">
                      <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="content-tit">${activity_data.activity_title}</a>
                    </div>
                    <div class="bottom-content-wrap">
                      <div class="bottom-content-ahead">
                        <span></span>
                      </div>
                      <div class="bottom-content-back">
                      </div>
                    </div>
                  </div>
                </div>
              </div>`
        }
        else if (activityEndDate >= currentDate && activity_data.club__member_id===parseInt(member_id) && activity_data.thumbnail_path !== ''){
            text += `
            <div class="activity-wrap">
                <div class="activity-box">
                  <div class="activity-img-box">
                    <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="activity-img-link">
                      <img
                        src="/upload/${activity_data.thumbnail_path}"
                        alt="ces올인원 패키지"
                        class="activity-img"
                      />
                    </a>
                    <div class="like-btn-container">
                      <button type="button" class="like-btn ${activity_data.id}" >
                        <span class="like-on like-none">
                          <svg xmlns="http://www.w3.org/2000/svg" class="like-img-container ${activity_data.id} ${ activity_data.status }" viewBox="0 0 20 20" fill="currentColor">
                            <path
                              fill-rule="evenodd"
                              d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z"
                              clip-rule="evenodd"
                              class="like-img-container ${activity_data.id} ${ activity_data.status }"
                            ></path>
                          </svg>
                        </span>
                        <span class="like-off">
                        <input type="hidden" name="is-like" value="${ activity_data.status } ">
                          <svg
                            id="like-target"
                            xmlns="http://www.w3.org/2000/svg"
                            class="unlike-img-container ${activity_data.id} ${ activity_data.status }"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              stroke-linecap="round"
                              stroke-linejoin="round"
                              stroke-width="2"
                              d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                            ></path>
                          </svg>
                        </span>
                      </button>
                    </div>
                  </div>
                  <div class="content-txt-box">
                    <div class="middle-txt">
                      <div>
                        <a class="txt-head-confirm" href="http://127.0.0.1:10000/member/activity/?activity_id=${activity_data.id}">활동수정</a>
                      </div>
                      <div>
                        <span class="txt-head">${changeDate(activity_data.activity_end)}</span>
                      </div>
                      <div class="middle-txt-online">
                      </div>
                    </div>
                    <div class="content-tit-wrap">
                      <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.id}" class="content-tit">${activity_data.activity_title}</a>
                    </div>
                    <div class="bottom-content-wrap">
                      <div class="bottom-content-ahead">
                        <span></span>
                      </div>
                      <div class="bottom-content-back">
                      </div>
                    </div>
                  </div>
                </div>
              </div>`
        }
        // 좋아요 누른거
        else if (activity_data.status === true && activity_data.activity__thumbnail_path===''){
            console.log(activity_data)
            text += `<div class="activity-wrap">
                        <div class="activity-box">
                          <div class="activity-img-box">
                            <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.activity_id}" class="activity-img-link">
                              <img
                                src="/static/public/web/images/logo/logo8.png"
                                alt="ces올인원 패키지"
                                class="activity-img"
                              />
                            </a>
                            <div class="like-btn-container">
                              <button type="button" class="like-btn ${activity_data.activity_id}" >
                                <span class="like-on like-none">
                                  <svg xmlns="http://www.w3.org/2000/svg" class="like-img-container ${activity_data.activity_id} ${ activity_data.status }" viewBox="0 0 20 20" fill="currentColor">
                                    <path
                                      fill-rule="evenodd"
                                      d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z"
                                      clip-rule="evenodd"
                                      class="like-img-container ${activity_data.activity_id} ${ activity_data.status }"
                                    ></path>
                                  </svg>
                                </span>
                                <span class="like-off">
                                <input type="hidden" name="is-like" value="${ activity_data.status } ">
                                  <svg
                                    id="like-target"
                                    xmlns="http://www.w3.org/2000/svg"
                                    class="unlike-img-container ${activity_data.activity_id} ${ activity_data.status }"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                  >
                                    <path
                                      stroke-linecap="round"
                                      stroke-linejoin="round"
                                      stroke-width="2"
                                      d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                                    ></path>
                                  </svg>
                                </span>
                              </button>
                            </div>
                          </div>
                          <div class="content-txt-box">
                            <div class="middle-txt">
                              <div>
                                <span class="txt-head-confirm">관심활동</span>
                              </div>
                              <div>
                                <span class="txt-head">${changeDate(activity_data.activity__activity_end)}</span>
                              </div>
                              <div class="middle-txt-online">
                              </div>
                            </div>
                            <div class="content-tit-wrap">
                              <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.activity_id}" class="content-tit">${activity_data.activity__activity_title}</a>
                            </div>
                            <div class="bottom-content-wrap">
                              <div class="bottom-content-ahead">
                                <span></span>
                              </div>
                              <div class="bottom-content-back">
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                        `
        }
        else if (activity_data.status === true && activity_data.activity__thumbnail_path !==''){
            console.log(activity_data)
            text += `<div class="activity-wrap">
                        <div class="activity-box">
                          <div class="activity-img-box">
                            <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.activity_id}" class="activity-img-link">
                              <img
                                src="/upload/${activity_data.activity__thumbnail_path}"
                                alt="ces올인원 패키지"
                                class="activity-img"
                              />
                            </a>
                            <div class="like-btn-container">
                              <button type="button" class="like-btn ${activity_data.activity_id}" >
                                <span class="like-on like-none">
                                  <svg xmlns="http://www.w3.org/2000/svg" class="like-img-container ${activity_data.activity_id} ${ activity_data.status }" viewBox="0 0 20 20" fill="currentColor">
                                    <path
                                      fill-rule="evenodd"
                                      d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z"
                                      clip-rule="evenodd"
                                      class="like-img-container ${activity_data.activity_id} ${ activity_data.status }"
                                    ></path>
                                  </svg>
                                </span>
                                <span class="like-off">
                                <input type="hidden" name="is-like" value="${ activity_data.status } ">
                                  <svg
                                    id="like-target"
                                    xmlns="http://www.w3.org/2000/svg"
                                    class="unlike-img-container ${activity_data.activity_id} ${ activity_data.status }"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                  >
                                    <path
                                      stroke-linecap="round"
                                      stroke-linejoin="round"
                                      stroke-width="2"
                                      d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                                    ></path>
                                  </svg>
                                </span>
                              </button>
                            </div>
                          </div>
                          <div class="content-txt-box">
                            <div class="middle-txt">
                              <div>
                                <span class="txt-head-confirm">관심활동</span>
                              </div>
                              <div>
                                <span class="txt-head">${changeDate(activity_data.activity__activity_end)}</span>
                              </div>
                              <div class="middle-txt-online">
                              </div>
                            </div>
                            <div class="content-tit-wrap">
                              <a href="http://127.0.0.1:10000/activity/detail/?id=${activity_data.activity_id}" class="content-tit">${activity_data.activity__activity_title}</a>
                            </div>
                            <div class="bottom-content-wrap">
                              <div class="bottom-content-ahead">
                                <span></span>
                              </div>
                              <div class="bottom-content-back">
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                        `
        }
    })
    changeLike()

}
return text;}


const changeLike = async ()=>{
    const conditions = document.querySelectorAll(".unlike-img-container")
    const on = document.querySelectorAll(".like-on")
    const off = document.querySelectorAll(".like-off")
    conditions.forEach((condition,i)=>{
        if(condition.classList[2] === "true"){
        on[i].classList.remove('like-none')
        off[i].classList.add('like-none')
    }
    })
}






inner.addEventListener("click", async(e)=> {
    if(e.target.classList[0] === 'unlike-img-container'){
        const likly = e.target.classList[1]
        await activityService.update(likly)
        await activityService.getList(member_id, page,status_like, showList).then((text) => {
        inner.innerHTML = text;
        changeLike()
});


    }
    if(e.target.classList[0] === 'like-img-container'){
        const likly = e.target.classList[1]
        await activityService.update(likly)
        await activityService.getList(member_id, page,status_like, showList).then((text) => {
        inner.innerHTML = text;
        changeLike()
});
    }

})



activityService.getList(member_id, page + 1).then((activity_data) => {
    if (activity_data.length !== 0){
        cansleButton.style.display = "flex";
        changeLike()
    }
});

activityService.getList(member_id, page + 1).then((activity_data) => {
    if (activity_data.length === 0){
        cansleButton.style.display = "none";
        changeLike()
    }
});


const cansleButton = document.getElementById("teenchin-more-btn")

cansleButton.addEventListener("click", (e) => {
    activityService.getList(member_id, ++page,status_like, showList).then((text) => {
        inner.innerHTML += text;
        changeLike()
    });

    activityService.getList(member_id, page + 1,).then((activity_data) => {
    if (activity_data.length === 0){
        cansleButton.style.display = "none";
        changeLike()
    }
});

});




activityService.getList(member_id, page,status_like, showList).then((text) => {
    inner.innerHTML += text;
    changeLike()
});




activityService.getList(member_id, page,status_like, showList);





const changeDate = (dateStr) => {
    const days = ['일', '월', '화', '수', '목', '금', '토'];
    const date = new Date(dateStr); // string타입을 date타입으로 바꿈.
    const month = date.getMonth() + 1; // getMonth()는 0부터 시작하므로 1을 더함.
    const day = date.getDate();
    const dayOfWeek = days[date.getDay()]; // 요일을 숫자로 가져와 해당 요일 문자열로 변환

    // padStart는 ,앞의 숫자의 자릿수 만큼 0을 채워줌
    return `${month.toString().padStart(2, '0')}월 ${day.toString().padStart(2, '0')}일(${dayOfWeek})`;
}

