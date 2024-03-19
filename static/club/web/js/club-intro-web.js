const banner = document.querySelector(".main-slide-swiper");
let count = 0;
let firstDiv = document.createElement("div");
let lastDiv = document.createElement("div");
firstDiv.setAttribute("class", "main-slide-content-wrap");
lastDiv.setAttribute("class", "main-slide-content-wrap");

// 첫 화면에서는 첫 번째 배너가 보이지만,
// 이 함수가 실행되는 순간 두 번째 배너로 넘어간다.
function autoSlide() {
    // 이동시간 0.8초
    banner.style.transition = "transform 0.8s";
    // 마지막 슬라이드일 때
    // 10번 뒤에 1번을 배치시킨다.
    // 10번에서 1번으로 슬라이드 진행
    // 0s를 줘서 원래 1번 위치로 이동(슬라이드 효과는 안 보임)
    count++;
    if (count == 10) {
        banner.style.transform =
            "translate(-" + (631 * (count + 1) - 67.5) + "px)";
        setTimeout(function () {
            banner.style.transition = "transform 0s";
            banner.style.transform = "translate(-563.5px)";
        }, 800);
        count = 0;
    } else {
        banner.style.transform =
            "translate(-" + (631 * (count + 1) - 67.5) + "px)";
    }
}

// 가장 마지막에 첫 번째 배너를 이어 붙인다.
firstDiv.innerHTML = `<div class="main-slide-content-container">
                            <img
                                src="https://event-us.kr/Content/neweventus/image/landing/hostcenter/service/marketing/solution_1.png"
                                alt="슬라이드이미지"
                                class="main-slide-content-img"
                            />
                            <div
                                class="main-slide-content-text-wrap"
                            >
                                <div
                                    class="main-slide-content-text"
                                >
                                    플랫폼 메인 배너
                                </div>
                                <!---->
                            </div>
                        </div>
                    </div>`;
banner.appendChild(firstDiv);

// 가장 첫 번째에 마지막 배너를 이어 붙인다.
lastDiv.innerHTML = `<div class="main-slide-content-container">
                            <img
                                src="https://event-us.kr/Content/neweventus/image/landing/hostcenter/service/marketing/solution_10.png"
                                alt="슬라이드이미지"
                                class="main-slide-content-img"
                            />
                            <div
                                class="main-slide-content-text-wrap"
                            >
                                <div
                                    class="main-slide-content-text"
                                >
                                    단독 EDM
                                </div>
                                <!---->
                            </div>
                        </div>
                    </div>`;
banner.insertBefore(lastDiv, document.getElementById("first-slide-banner"));

// 첫 번째 배너는 10번이니까 왼쪽으로 한 번 밀어서 1번이 보이게 한다.
banner.style.transform = "translate(-563.5px)";

// 4초마다 슬라이드 이동
let inter = setInterval(autoSlide, 4000);

// 이전 버튼, 다음 버튼 구현
const arrows = document.querySelectorAll(".slide-arrow");
let arrowButtonCheck = true;
arrows.forEach((arrow) => {
    arrow.addEventListener("click", function () {
        if (arrowButtonCheck) {
            arrowButtonCheck = false;
            clearInterval(inter);
            banner.style.transition = "transform 0.8s";
            let arrowType = arrow.classList[0];
            if (arrowType === "slide-left-arrow") {
                count--;
                if (count == -1) {
                    banner.style.transform = "translate(67.5px)";
                    setTimeout(function () {
                        banner.style.transition = "transform 0s";
                        banner.style.transform = "translate(-6242.5px)";
                    }, 800);
                    count = 9;
                } else {
                    banner.style.transform =
                        "translate(-" + (631 * (count + 1) - 67.5) + "px)";
                }
            } else {
                count++;
                if (count == 10) {
                    banner.style.transform =
                        "translate(-" + (631 * (count + 1) - 67.5) + "px)";
                    setTimeout(function () {
                        banner.style.transition = "transform 0s";
                        banner.style.transform = "translate(-563.5px)";
                    }, 800);
                    count = 0;
                } else {
                    banner.style.transform =
                        "translate(-" + (631 * (count + 1) - 67.5) + "px)";
                }
            }
            inter = setInterval(autoSlide, 4000);
            setTimeout(function () {
                arrowButtonCheck = true;
            }, 800);
        }
    });
});
