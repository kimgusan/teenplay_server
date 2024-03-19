NodeList.prototype.filter = Array.prototype.filter;

const pauseIcons = document.querySelectorAll(".pause");
const playIcons = document.querySelectorAll(".restart");
const videos = document.querySelectorAll(".play-video");
const videoInfos = document.querySelectorAll(".play-info-wrap");
const controlButtons = document.querySelectorAll(".play-control-wrap");
const progressBars = document.querySelectorAll(".progress-bar-now");
const muteIcons = document.querySelectorAll(".mute");
const unmuteIcons = document.querySelectorAll(".unmute");
const videoWraps = document.querySelectorAll(".play-each");
let nowPlaying = document.querySelector(".play-each.playing");
const slideWrap = document.querySelector(".play-items");
const slideContainer = document.querySelector(".play-item");
const likeBtns = document.querySelectorAll(".play-like-btn");
const emptyHeart = document.querySelectorAll(".play-like-icon.empty");
const fullHeart = document.querySelectorAll(".play-like-icon.full");

// 재생 중이 아닌 영상은 일시정지로 시작

videoWraps.forEach((videoWrap, i) => {
    if (!videoWrap.classList.contains("playing")) {
        videos[i].autoplay = false;
    } else {
        videos[i].autoplay = true;
    }
});

videos[0].play();

// 일시정지, 재생 관련 버튼
// false일 때 클릭 시 재생, true일 때 클릭 시 일시정지
globalThis.flags = new Array(videos.length);

videos.forEach((video, i) => {
    video.addEventListener("click", (e) => {
        if (!globalThis.flags[i]) {
            globalThis.flags[i] = true;
            pauseIcons[i].style.display = "none";
            playIcons[i].style.display = "block";
            e.target.pause();
        } else {
            globalThis.flags[i] = false;
            pauseIcons[i].style.display = "block";
            playIcons[i].style.display = "none";
            e.target.play();
        }
    });
});

muteIcons.forEach((mute) => (mute.style.display = "none"));
unmuteIcons.forEach((unmute) => (unmute.style.display = "block"));

// 음소거 관련 버튼
muteIcons.forEach((mute) => {
    mute.addEventListener("click", () => {
        muteIcons.forEach((mute) => {
            mute.style.display = "none";
        });
        videos.forEach((video) => {
            video.muted = true;
        });
        unmuteIcons.forEach((unmute) => {
            unmute.style.display = "block";
        });
    });
});

unmuteIcons.forEach((unmute) => {
    unmute.addEventListener("click", (e) => {
        unmuteIcons.forEach((unmute) => {
            unmute.style.display = "none";
        });
        videos.forEach((video) => {
            video.muted = false;
        });
        muteIcons.forEach((mute) => {
            mute.style.display = "block";
        });
    });
});

// 진행도 1초마다 증가
videos.forEach((video, i) => {
    video.addEventListener("timeupdate", (e) => {
        let percent = (e.target.currentTime / e.target.duration) * 100;
        progressBars[i].style.width = `${percent}%`;
    });
});

// 스크롤로 이전/다음 틴플레이 이동

function slideNext(idx) {
    slideContainer.style.transition = `all 0.5s ease-in`;
    slideContainer.style.transform = `translateY(calc(-1 * (100vh * ${idx}))`;
    videoWraps[idx - 1].classList.remove("playing");
    videoWraps[idx].classList.add("playing");
}

function slidePrev(idx) {
    slideContainer.style.transition = `all 0.5s ease-in`;
    slideContainer.style.transform = `translateY(calc(-1 * (100vh * ${idx}))`;
    videoWraps[idx + 1].classList.remove("playing");
    videoWraps[idx].classList.add("playing");
}

function manageScroll(e) {
    e.preventDefault();
    e.stopPropagation();
    return;
}

let idx = 0;
let check = true;
slideWrap.addEventListener("wheel", (e) => {
    manageScroll(e);
    if (!check) return;
    check = false;
    for (let i = 0; i < videoWraps.length; i++) {
        if (videoWraps[i].classList.contains("playing")) {
            idx = i;
            break;
        }
    }
    if (e.deltaY > 0) {
        setTimeout(() => {
            check = true;
        }, 800);
        if (idx == videoWraps.length - 1) {
            return;
        }
        slideNext(idx + 1);
        videos[idx].pause();
        globalThis.flags[idx] = true;
        pauseIcons[idx].style.display = "none";
        playIcons[idx].style.display = "block";
        idx++;
        videos[idx].play();
        globalThis.flags[idx] = false;
        pauseIcons[idx].style.display = "block";
        playIcons[idx].style.display = "none";
    } else {
        setTimeout(() => {
            check = true;
        }, 800);
        if (idx == 0) {
            return;
        }
        slidePrev(idx - 1);
        videos[idx].pause();
        globalThis.flags[idx] = true;
        pauseIcons[idx].style.display = "none";
        playIcons[idx].style.display = "block";
        idx--;
        videos[idx].play();
        globalThis.flags[idx] = false;
        pauseIcons[idx].style.display = "block";
        playIcons[idx].style.display = "none";
    }
});

// 터치로 이동시키기

// [html 최초 로드 및 이벤트 상시 대기 실시]
window.onload = () => {
    main();
};

function main() {
    slideWrap.addEventListener("touchstart", handleStart, false);
    slideWrap.addEventListener("touchmove", handleMove, false);
    slideWrap.addEventListener("touchend", handleEnd, false);

    // 모바일 : 터치 시작 내부 함수
    function handleStart(e) {
        // body 스크롤 막음
        BodyScrollDisAble(e);

        // 터치한 요소의 id값 확인
        // let startId = e.targetTouches[0].target.id;
        // console.log(`startId: ${startId}`);
        // console.log("");

        // 좌표값 확인
        // let startX = e.changedTouches[0].clientX;
        // let startY = e.changedTouches[0].clientY;
        // console.log(`startX: ${startX}`);
        // console.log(`startY: ${startY}`);
        // console.log("");
    }

    // 모바일 : 터치 이동 내부 함수
    function handleMove(e) {
        // body 스크롤 막음
        BodyScrollDisAble(e);

        // 터치한 요소의 id값 확인
        // let moveId = e.targetTouches[0].target.id;
        // console.log(`moveId: ${moveId}`);
        // console.log("");

        // 좌표값 확인
        // let moveX = e.changedTouches[0].clientX;
        // let moveY = e.changedTouches[0].clientY;
        // console.log(`moveX: ${moveX}`);
        // console.log(`moveY: ${moveY}`);
        // console.log("");
    }

    // 모바일 : 터치 종료 내부 함수
    function handleEnd(e) {
        // body 스크롤 허용
        BodyScrollDisAble(e);

        // 터치한 요소의 id값 확인
        // let moveId = e.targetTouches[0].target.id;
        // console.log(`moveId: ${moveId}`);
        // console.log("");

        // 좌표값 확인
        // let endX = e.changedTouches[0].clientX;
        // let endY = e.changedTouches[0].clientY;
        // console.log(`endX: ${endX}`);
        // console.log(`endY: ${endY}`);
        // console.log("");
    }

    // 스크롤 막기
    function BodyScrollDisAble() {
        slideWrap.style.overflow = "hidden";
    }

    // 스크롤 허용
    function BodyScrollAble() {
        slideWrap.style.overflow = "auto";
    }
}

// 스와이프 방향 알아내기
let initialX = null,
    initialY = null;

function initTouch(e) {
    initialX = `${e.touches ? e.touches[0].clientX : e.clientX}`;
    initialY = `${e.touches ? e.touches[0].clientY : e.clientY}`;
}

let direction = 0;
let touchIdx = 0;
let touchCheck = true;

function swipeDirection(e) {
    if (initialX !== null && initialY !== null) {
        const currentX = `${e.touches ? e.touches[0].clientX : e.clientX}`;
        const currentY = `${e.touches ? e.touches[0].clientY : e.clientY}`;

        let diffX = initialX - currentX;
        let diffY = initialY - currentY;

        Math.abs(diffX) <= Math.abs(diffY) && (0 < diffY ? (direction = 1) : (direction = -1));
        // console.log(direction);
        initialX = null;
        initialY = null;

        if (!touchCheck) return;
        touchCheck = false;
        for (let i = 0; i < videoWraps.length; i++) {
            if (videoWraps[i].classList.contains("playing")) {
                touchIdx = i;
                break;
            }
        }
        if (direction > 0) {
            setTimeout(() => {
                touchCheck = true;
            }, 800);
            if (touchIdx == videoWraps.length - 1) {
                return;
            }
            slideNext(touchIdx + 1);
            videos[touchIdx].pause();
            globalThis.flags[touchIdx] = true;
            pauseIcons[touchIdx].style.display = "none";
            playIcons[touchIdx].style.display = "block";
            touchIdx++;
            videos[touchIdx].play();
            globalThis.flags[touchIdx] = false;
            pauseIcons[touchIdx].style.display = "block";
            playIcons[touchIdx].style.display = "none";
        } else {
            setTimeout(() => {
                touchCheck = true;
            }, 800);
            if (touchIdx == 0) {
                return;
            }
            slidePrev(touchIdx - 1);
            videos[touchIdx].pause();
            globalThis.flags[touchIdx] = true;
            pauseIcons[touchIdx].style.display = "none";
            playIcons[touchIdx].style.display = "block";
            touchIdx--;
            videos[touchIdx].play();
            globalThis.flags[touchIdx] = false;
            pauseIcons[touchIdx].style.display = "block";
            playIcons[touchIdx].style.display = "none";
        }
    }
}

slideWrap.addEventListener("touchstart", initTouch);
slideWrap.addEventListener("touchmove", swipeDirection);
// slideWrap.addEventListener("mousedown", (e) => {
//     initTouch(e), slideWrap.addEventListener("mousemove", swipeDirection);
// });
slideWrap.addEventListener("mouseup", () => {
    slideWrap.removeEventListener("mousemove", swipeDirection);
});

slideWrap.addEventListener("mousedown", (e) => {
    0 === e.button && (initTouch(e), slideWrap.addEventListener("mousemove", swipeDirection));
});

// 좋아요 아이콘 클릭 시 반영
likeBtns.forEach((button, i) => {
    button.addEventListener("click", () => {
        if (emptyHeart[i].style.display == "none") {
            emptyHeart[i].style.display = "block";
            fullHeart[i].style.display = "none";
        } else {
            emptyHeart[i].style.display = "none";
            fullHeart[i].style.display = "block";
        }
    });
});

// 좋아요 수 증가는 비동기로 좋아요 db에 반영 후 가져와 넣기 때문에
// 현재 화면에서는 구현하지 않습니다.
