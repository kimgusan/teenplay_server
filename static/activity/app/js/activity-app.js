// (일시) 라디오 버튼을 모두 돌면서 1개만 선택하도록 체크
document.querySelectorAll(".content-date .radio-everyday").forEach(function (radio) {
    radio.addEventListener("change", function () {
        // > 모든 라디오 버튼을 돌면서 현재 선택된 것 이외의 것들을 언체크
        document.querySelectorAll(".content-date .radio-everyday").forEach(function (otherRadio) {
            if (otherRadio !== radio) {
                otherRadio.checked = false;
                otherRadio.parentNode.nextElementSibling.style.display = "none"; // 다른 라디오 버튼의 텍스트
            }
        });
    });
});

// (참여 방법)라디오 버튼을 모두 돌면서 1개만 선택하도록 체크
document.querySelectorAll(".content-method .radio-everyday").forEach(function (radio) {
    radio.addEventListener("change", function () {
        // > 모든 라디오 버튼을 돌면서 현재 선택된 것 이외의 것들을 언체크
        document.querySelectorAll(".content-method .radio-everyday").forEach(function (otherRadio) {
            if (otherRadio !== radio) {
                otherRadio.checked = false;
                otherRadio.parentNode.nextElementSibling.style.display = "none"; // 다른 라디오 버튼의 텍스트 감춤
            }
        });
    });
});

// (가격) 라디오 버튼을 모두 돌면서 1개만 선택하도록 체크
document.querySelectorAll(".content-method-pay .radio-everyday").forEach(function (radio) {
    radio.addEventListener("change", function () {
        // > 모든 라디오 버튼을 돌면서 현재 선택된 것 이외의 것들을 언체크
        document.querySelectorAll(".content-method-pay .radio-everyday").forEach(function (otherRadio) {
            if (otherRadio !== radio) {
                otherRadio.checked = false;
                otherRadio.parentNode.nextElementSibling.style.display = "none"; // 다른 라디오 버튼의 텍스트 감춤
            }
        });
    });
});
const activeLikeBtnsCopy = document.querySelectorAll(".like-btn-shadow");
const emptyHeartsCopy = document.querySelectorAll(".like-btn-shadow .like-btn-display");
const fullHeartsCopy = document.querySelectorAll(".like-btn-shadow .like-btn-display-none");

// 기간선택 체크 시 기간 달력 display 유무 표시
const checkedDate = document.getElementById("ev-radio-102");

const dateRangeCheck = document.querySelectorAll(".date-container .date-box .radio-everyday");
dateRangeCheck.forEach((selectDateRange) => {
    selectDateRange.addEventListener("click", (e) => {
        if (selectDateRange.id == "ev-radio-102" && selectDateRange.checked == true) {
            dateRange = document.querySelector(".date-display-none");
            dateRange.style.display = "block";
        } else {
            dateRange.style.display = "none";
        }
    });
});

// checkedDate.addEventListener("change", (e) => {
//     dateRange = document.querySelector(".date-display-none");
//     dateRange.style.display = "block";

// if (e.target.id == "ev-radio-102") {
//     dateRange.style.display = "block";
//     // > 달력 선택 시 창이 사라지지 않게 만드는 용도
// } else if (e.target.classList == "relative-date") {
//     console.log(e.target.classList);
//     dateRange.style.display = "block";
// } else {
//     dateRange.style.display = "none";
// }
// });

// 좋아요 선택 시 동작하는 js
const modalWrapCopy = document.querySelector(".club-modal-wrap");
// > 좋아요 선택 시 관심 설정할 때 표시할 부분
const modalLikeContainerCopy = document.querySelector(".club-modal-like-wrap:not(.unlike)");
// > 좋아요 선택 시 관심 해제할 때 표시할 부분
const modalUnlikeContainerCopy = document.querySelector(".club-modal-like-wrap.unlike");

activeLikeBtnsCopy.forEach((button, i) => {
    button.addEventListener("click", () => {
        modalWrapCopy.style.display = "block";
        if (emptyHeartsCopy[i].style.display === "none") {
            modalUnlikeContainerCopy.style.display = "block";
            modalLikeContainerCopy.style.display = "none";
            emptyHeartsCopy[i].style.display = "block";
            fullHeartsCopy[i].style.display = "none";
        } else {
            modalUnlikeContainerCopy.style.display = "none";
            modalLikeContainerCopy.style.display = "block";
            emptyHeartsCopy[i].style.display = "none";
            fullHeartsCopy[i].style.display = "block";
        }
    });
});

// 좋아요 선택 시 창 내 버튼 클릭 시 모달창 닫기
const modalLikeExitBtnCopy = document.querySelector(".club-modal-like-button");
const modalUnlikeExitBtnCopy = document.querySelector(".club-modal-unlike-button");

function exitModal() {
    modalWrapCopy.style.display = "none";
}

modalLikeExitBtnCopy.addEventListener("click", exitModal);
modalUnlikeExitBtnCopy.addEventListener("click", exitModal);

// 추천순을 선택 했을 때 bold 부분
const buttons = document.querySelectorAll("button[name='selected-recommend-btn']");
const filterSpans = document.querySelectorAll("span.filter-span");

buttons.forEach((button, i) => {
    button.addEventListener("click", () => {
        filterSpans.forEach((span, j) => {
            if (i == j && !span.classList.contains("font-bold-choice")) {
                span.classList.add("font-bold-choice");
            } else if (i != j && span.classList.contains("font-bold-choice")) {
                span.classList.remove("font-bold-choice");
            }
        });
    });
});

//  radio 버튼 클릭 된 값을 가져와서 hidden box의 명칭을 변경하기
let radioCheck = document.querySelectorAll(".date-box .radio-everyday");

radioCheck.forEach((radio) => {
    radio.addEventListener("click", (e) => {
        // 클릭한 위치의 value 값을 div의 text 명칭으로 변경하기
        let dateTextRadio = document.querySelector(".find-filter-bg");
        let dateText = document.getElementById("date-text");
        if (e.target.value == "모든날") {
            dateText.innerHTML = e.target.value;
            dateTextRadio.style.display = "none";
        } else {
            dateText.innerHTML = e.target.value;
            dateTextRadio.style.display = "flex";
        }
    });
});

// 참여방법
let joinCheck = document.querySelectorAll(".method-box .radio-everyday");

joinCheck.forEach((join) => {
    join.addEventListener("click", (e) => {
        // 클릭한 위치의 value 값을 div의 text 명칭으로 변경하기
        let joinTextRadio = document.querySelector(".find-filter-bg-join");
        let joinText = document.getElementById("join-content");
        if (e.target.value == "전체") {
            joinText.innerHTML = e.target.value;
            joinTextRadio.style.display = "none";
        } else {
            joinText.innerHTML = e.target.value;
            joinTextRadio.style.display = "flex";
        }
    });
});

// 가격
let payCheck = document.querySelectorAll(".method-box-pay .radio-everyday");

payCheck.forEach((pay) => {
    pay.addEventListener("click", (e) => {
        // 클릭한 위치의 value 값을 div의 text 명칭으로 변경하기
        let payTextRadio = document.querySelector(".find-filter-bg-pay");
        let payText = document.getElementById("pay-content");
        if (e.target.value == "") {
            payText.innerHTML = e.target.value;
            payTextRadio.style.display = "none";
        } else {
            payText.innerHTML = e.target.value;
            payTextRadio.style.display = "flex";
        }
    });
});

// 지역
let blockLocation = document.querySelector(".block-location");
blockLocation.addEventListener("change", (e) => {
    let locationTextRadio = document.querySelector(".find-filter-bg-location");
    let locationText = document.getElementById("location-content");
    if (e.target.value == "전체") {
        locationText.innerHTML = e.target.value;
        locationTextRadio.style.display = "none";
    } else {
        locationText.innerHTML = e.target.value;
        locationTextRadio.style.display = "flex";
    }
});

// 체크박스를 선택했을 때 div 추가되는 구문
let endDateTie = document.querySelector(".end-date-tie");
endDateTie.addEventListener("click", (e) => {
    if (e.target.tagName == "INPUT") {
        if (e.target.checked) {
            let hiddenBg = document.querySelector(".hidden-filter-container-check");
            hiddenBg.innerHTML += `<div class="find-filter-bg-two" id="date-end-activity style="display: flex" value="${e.target.value}">
                                        <div id="date-text">${e.target.value}</div>
                                        <button class="find-filter-exit">
                                            <svg class="exit-icon-button" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" id="exit-icon-button">
                                                <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                                            </svg>
                                        </button>
                                    </div>`;
        }
    }
});

// 반대로 해당 버튼을 클릭했을 때 체크가 풀리는 부분을 만들어 줌
endDateTie.addEventListener("click", (e) => {
    if (e.target.tagName == "INPUT") {
        if (!e.target.checked) {
            let values = e.target.value;
            let divToRemove = document.querySelector(".find-filter-bg-two[value='" + values + "']");
            if (divToRemove) {
                divToRemove.remove();
            }
            // divToRemove.parentNode.removeChild(divToRemove);
            // 위에서 div 생성 시 id값으로 생성해 줬기 때문에 해당 id 값을 찾아서 div를 지워주면 된다
        }
    }
});

// 버튼을 클릭했을 때 행사분야 display(checkbox) 선택해서 보여주기 위한 div 추가
// > 행사분야의 체크박스를 선택했을 떄 나오는 input.value값을 알기위한 변수
let actCategoryCenter = document.querySelectorAll(".activity-center");

// > 행사 분야 내부에 여러 카테고리 중 클릭이 가능한 부분이 있는지 확인을 위한 반복 구문 사용
actCategoryCenter.forEach((actCategory) => {
    // > 해당 행사 구문에서 반복을 돌리던 중 check이벤트가 발생 시
    actCategory.addEventListener("click", (e) => {
        // > 해당 이벤트의 target의 태그 이름이 input일 때
        if (e.target.tagName == "INPUT") {
            // > input의 checked 가 true로 되었을 경우
            if (e.target.checked) {
                // > 숨겨진 항목을 표기하는 hidden div 내부에 target 의 value 값을 추가하여
                // > 특정 HTML을 추가 한디
                let hiddenBg = document.querySelector(".hidden-filter-container-check");
                hiddenBg.innerHTML += `<div class="find-filter-bg-three" id="date-text-f-category style="display: flex" value="${e.target.value}">
                                        <div id="date-text">${e.target.value}</div>
                                        <button class="find-filter-exit">
                                            <svg class="exit-icon-button" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" id="exit-icon-button">
                                                <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                                            </svg>
                                        </button>
                                    </div>`;
            }
        }
    });
});
// 반대로 해당 버튼을 클릭했을 때 체크가 풀리는 부분을 만들어 줌
actCategoryCenter.forEach((actCategory) => {
    actCategory.addEventListener("click", (e) => {
        if (e.target.tagName == "INPUT") {
            if (!e.target.checked) {
                let values = e.target.value;
                let divToRemove = document.querySelector(".find-filter-bg-three[value='" + values + "']");
                if (divToRemove) {
                    divToRemove.remove();
                }
                // divToRemove.parentNode.removeChild(divToRemove);
                // 위에서 div 생성 시 id값으로 생성해 줬기 때문에 해당 id 값을 찾아서 div를 지워주면 된다
            }
        }
    });
});

let actCategoryCenterContent = document.querySelectorAll(".act-category-center");

// > 행사유형 내부에 여러 카테고리 중 클릭이 가능한 부분이 있는지 확인을 위한 반복 구문 사용
actCategoryCenterContent.forEach((actCategory) => {
    // > 해당 행사 구문에서 반복을 돌리던 중 check이벤트가 발생 시
    actCategory.addEventListener("click", (e) => {
        // > 해당 이벤트의 target의 태그 이름이 input일 때
        if (e.target.tagName == "INPUT") {
            // > input의 checked 가 true로 되었을 경우
            if (e.target.checked) {
                // > 숨겨진 항목을 표기하는 hidden div 내부에 target 의 value 값을 추가하여
                // > 특정 HTML을 추가 한디
                let hiddenBg = document.querySelector(".hidden-filter-container-check");
                hiddenBg.innerHTML += `<div class="find-filter-bg-three" id="date-text-f-category style="display: flex" value="${e.target.value}">
                                        <div id="date-text">${e.target.value}</div>
                                        <button class="find-filter-exit">
                                            <svg class="exit-icon-button" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" id="exit-icon-button">
                                                <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                                            </svg>
                                        </button>
                                    </div>`;
            }
        }
    });
});
// 반대로 해당 버튼을 클릭했을 때 체크가 풀리는 부분을 만들어 줌
actCategoryCenterContent.forEach((actCategory) => {
    actCategory.addEventListener("click", (e) => {
        if (e.target.tagName == "INPUT") {
            if (!e.target.checked) {
                let values = e.target.value;
                let divToRemove = document.querySelector(".find-filter-bg-three[value='" + values + "']");
                if (divToRemove) {
                    divToRemove.remove();
                }
                // divToRemove.parentNode.removeChild(divToRemove);
                // 위에서 div 생성 시 id값으로 생성해 줬기 때문에 해당 id 값을 찾아서 div를 지워주면 된다
            }
        }
    });
});

// 필터 타이틀에서 x 버튼 클릭 시 display 가 none으로 변경되고 초기 상태로 돌아가는 기능
let checkExit = document.getElementById("hidden-filter-container-check");

checkExit.addEventListener("click", (e) => {
    if (e.target.classList.contains("exit-icon-button")) {
        let parentDiv = e.target.closest(".find-filter-bg");
        if (parentDiv) {
            parentDiv.style.display = "none";
            document.querySelectorAll(".content-date .radio-everyday").forEach(function (otherRadio) {
                if (otherRadio.id == "ev-radio-98") {
                    otherRadio.checked = true;
                } else {
                    otherRadio.checked = false;
                }
            });
        }
    }
});

checkExit.addEventListener("click", (e) => {
    if (e.target.classList.contains("exit-icon-button")) {
        let parentDiv = e.target.closest(".find-filter-bg-join");
        if (parentDiv) {
            parentDiv.style.display = "none";
            document.querySelectorAll(".content-method .radio-everyday").forEach(function (otherRadio) {
                if (otherRadio.id == "ev-radio-115") {
                    otherRadio.checked = true;
                } else {
                    otherRadio.checked = false;
                }
            });
        }
    }
});

checkExit.addEventListener("click", (e) => {
    if (e.target.classList.contains("exit-icon-button")) {
        let parentDiv = e.target.closest(".find-filter-bg-pay");
        if (parentDiv) {
            parentDiv.style.display = "none";
            document.querySelectorAll(".content-method-pay .radio-everyday").forEach(function (otherRadio) {
                if (otherRadio.id == "ev-radio-118") {
                    otherRadio.checked = true;
                } else {
                    otherRadio.checked = false;
                }
            });
        }
    }
});

// 지역
checkExit.addEventListener("click", (e) => {
    if (e.target.classList.contains("exit-icon-button")) {
        let parentDiv = e.target.closest(".find-filter-bg-location");
        if (parentDiv) {
            parentDiv.style.display = "none";
            // 셀렉트 박스 요소를 가져옵니다.
            const selectElement = document.querySelector(".block-location");

            // 첫 번째 옵션을 기본(default) 값으로 선택합니다.
            selectElement.selectedIndex = 0;
        }
    }
});

// 모집 종료 행사
let endDateCheck = document.querySelectorAll(".filter-space .end-date-tie");
endDateCheck.forEach((textCategoryValue) => {
    document.addEventListener("click", (e) => {
        let divTest = e.target.closest("div");
        var checkedValue = divTest.getAttribute("value");
        // check가 되어있는 것 중에 값이 값이 같은게 있으면 삭제
        let chkBtn = document.querySelectorAll(".filter-space .end-date-tie .end-date-chk");
        if (textCategoryValue.innerText == checkedValue) {
            chkBtn.forEach((inputName) => {
                let divToRemove = document.querySelector(".find-filter-bg-two[value='" + checkedValue + "']");
                if (divToRemove) {
                    divToRemove.remove();
                }
                if (inputName.getAttribute("value") == checkedValue) {
                    inputName.checked = false;
                }
            });
        }
    });
});

// 행사분야 체크박스 선택 시 삭제되는 기능

let allCategory = document.querySelectorAll(".filter-space .activity-center");
allCategory.forEach((textCategoryValue) => {
    document.addEventListener("click", (e) => {
        let divTest = e.target.closest("div");
        var checkedValue = divTest.getAttribute("value");
        // check가 되어있는 것 중에 값이 값이 같은게 있으면 삭제
        let chkBtn = document.querySelectorAll(".filter-space .activity-center .activity-chk");
        if (textCategoryValue.innerText == checkedValue) {
            chkBtn.forEach((inputName) => {
                let divToRemove = document.querySelector(".find-filter-bg-three[value='" + checkedValue + "']");
                if (divToRemove) {
                    divToRemove.remove();
                }
                if (inputName.getAttribute("value") == checkedValue) {
                    inputName.checked = false;
                }
            });
        }
    });
});

// 행사유형 체크박스 선택 시 삭제되는 기능
let allActCategory = document.querySelectorAll(".filter-space .act-category-center");
allActCategory.forEach((textCategoryValue) => {
    document.addEventListener("click", (e) => {
        let divTest = e.target.closest("div");
        var checkedValue = divTest.getAttribute("value");
        // check가 되어있는 것 중에 값이 값이 같은게 있으면 삭제
        let chkBtn = document.querySelectorAll(".filter-space .act-category-center .act-category-chk");
        if (textCategoryValue.innerText == checkedValue) {
            chkBtn.forEach((inputName) => {
                let divToRemove = document.querySelector(".find-filter-bg-three[value='" + checkedValue + "']");
                if (divToRemove) {
                    divToRemove.remove();
                }
                if (inputName.getAttribute("value") == checkedValue) {
                    inputName.checked = false;
                }
            });
        }
    });
});

// 핕터 초기화
let filterReset = document.querySelector(".filter-reset");
filterReset.addEventListener("click", (e) => {
    // 달력 초기화
    dateRange = document.querySelector(".date-display-none");
    dateRange.style.display = "none";

    // 일시 초기화
    let dateTextRadio = document.querySelector(".find-filter-bg");
    let dateText = document.getElementById("date-text");
    dateTextRadio.style.display = "none";
    dateText.innerHTML = "모든날";
    document.querySelectorAll(".content-date .radio-everyday").forEach(function (otherRadio) {
        if (otherRadio.id == "ev-radio-98") {
            otherRadio.checked = true;
        } else {
            otherRadio.checked = false;
        }
        // 참여방법 초기화
        let joinTextRadio = document.querySelector(".find-filter-bg-join");
        let joinText = document.getElementById("join-content");
        joinTextRadio.style.display = "none";
        joinText.innerHTML = "전체";
        document.querySelectorAll(".content-method .radio-everyday").forEach(function (otherRadio) {
            if (otherRadio.id == "ev-radio-115") {
                otherRadio.checked = true;
            } else {
                otherRadio.checked = false;
            }
        });

        // 가격 초기화
        let payTextRadio = document.querySelector(".find-filter-bg-pay");
        let payText = document.getElementById("pay-content");
        payTextRadio.style.display = "none";
        payText.innerHTML = "";

        document.querySelectorAll(".content-method-pay .radio-everyday").forEach(function (otherRadio) {
            if (otherRadio.id == "ev-radio-118") {
                otherRadio.checked = true;
            } else {
                otherRadio.checked = false;
            }
        });
    });

    // 모집 종료 행사 필터에 선택되어 보여지는 태그 삭제
    let divToJoinRemove = document.querySelectorAll(".find-filter-bg-two");
    divToJoinRemove.forEach((names) => {
        names.remove();
    });

    //  모집 종료 행사 초기화
    let selectedJoinElements = document.querySelectorAll(".filter-space .end-date-tie .end-date-chk");
    // 각 체크된 요소에 대해 반복하여 체크를 해제합니다.
    selectedJoinElements.forEach((checkbox) => {
        checkbox.checked = false;
    });

    // 필터에 선택되어 보여지는 태그 삭제
    let divToRemove = document.querySelectorAll(".find-filter-bg-three");
    divToRemove.forEach((names) => {
        names.remove();
    });

    // 행사 분야 초기화
    let selectedElementsActivity = document.querySelectorAll(".filter-space .activity-center .activity-chk");
    selectedElementsActivity.forEach((checkbox) => {
        checkbox.checked = false;
    });

    // 행사 유형 초기화
    let selectedElements = document.querySelectorAll(".filter-space .act-category-center .act-category-chk");
    selectedElements.forEach((checkbox) => {
        checkbox.checked = false;
    });

    // 지역 초기화
    let locationTextRadio = document.querySelector(".find-filter-bg-location");
    let locationText = document.getElementById("location-content");
    locationTextRadio.style.display = "none";
    locationText.innerHTML = "";

    const selectElement = document.querySelector(".block-location");
    // 첫 번째 옵션을 기본(default) 값으로 선택합니다.
    selectElement.selectedIndex = 0;
});

// 접은 부분 피기 / 접기 (행사 분야)
let noHide = document.querySelector(".view-btn-tie");
let hide = document.querySelector(".view-btn-tie.hidden");
let hideButton = document.querySelector(".another-view-btn");

hideButton.addEventListener("click", (e) => {
    let closestText = e.target.closest("span");
    if (closestText.getAttribute("value") == "no-hidden") {
        noHide.classList.add("hidden");
        // 내용부분 표기
        let boxHidden = document.querySelectorAll(".activity-box-hidden");
        boxHidden.forEach((content) => {
            content.style.display = "flex";
        });
        hide.classList.remove("hidden");
    } else {
        hide.classList.add("hidden");
        let boxHidden = document.querySelectorAll(".activity-box-hidden");
        boxHidden.forEach((content) => {
            content.style.display = "none";
        });
        noHide.classList.remove("hidden");
    }
});

// 접은 부분 피기 / 접기 (행사 유형)
let noHideContent = document.querySelector(".view-btn-tie-two");
let hideContent = document.querySelector(".view-btn-tie-two.hidden");
let hideContentButton = document.querySelector(".another-view-btn-two");

hideContentButton.addEventListener("click", (e) => {
    let closestText = e.target.closest("span");
    if (closestText.getAttribute("value") == "no-hidden") {
        noHideContent.classList.add("hidden");
        // 내용부분 표기
        let boxHidden = document.querySelectorAll(".act-category-box-hidden");
        boxHidden.forEach((content) => {
            content.style.display = "flex";
        });
        hideContent.classList.remove("hidden");
    } else {
        hideContent.classList.add("hidden");
        let boxHidden = document.querySelectorAll(".act-category-box-hidden");
        boxHidden.forEach((content) => {
            content.style.display = "none";
        });
        noHideContent.classList.remove("hidden");
    }
});

// 채널 구독 선택 시 채널 구독중 으로 변경
// 채널 구독 시 채널 변경하기

let subBtn = document.querySelectorAll(".channel-click-btn-bg");
let channelBox = document.querySelectorAll(".channel-img-box");

subBtn.forEach((allBtn, i) => {
    allBtn.addEventListener("click", (e) => {
        channelBox.forEach((cBox, j) => {
            if (i == j) {
                let subModal = document.querySelectorAll(".subcribe-wrap");
                let channelSubModal = document.querySelectorAll(".channel-subcribe-m");
                subModal.forEach((subBox) => {
                    subBox.style.display = "flex";
                    subBtn[i].style.display = "none";
                    channelSubModal.forEach((ChannelSubBox, k) => {
                        if (i == k) {
                            ChannelSubBox.style.display = "flex";
                        }
                    });
                });
            }
        });
    });
});

// 반복을 사용해서 특정 인덱스 번호가 동일한 경우 사용하기 위한
let subBtnCopy = document.querySelectorAll(".channel-sub-m-btn");
let channelBoxCopy = document.querySelectorAll(".channel-img-box");
let test1 = document.querySelectorAll(".subcribe-wrap-cancel");
let test2 = document.querySelectorAll(".channel-subcribe-m");

// 구독 취소 선택 시 모달 뜨고 기존 상태로 변경
subBtnCopy.forEach((allBtnCopy, i) => {
    allBtnCopy.addEventListener("click", (e) => {
        channelBoxCopy.forEach((cBox, j) => {
            if (i == j) {
                test1.forEach((subBox) => {
                    subBox.style.display = "flex";
                    subBtn[i].style.display = "flex";
                    test2.forEach((ChannelSubBox, k) => {
                        if (i == k) {
                            ChannelSubBox.style.display = "none";
                        }
                    });
                });
            }
        });
    });
});

// 채널 구독 선택 시 나오는 모달창 끄기
document.querySelector(".meeting-continue-btn").addEventListener("click", (e) => {
    let subModal = document.querySelector(".subcribe-wrap");
    subModal.style.display = "none";
});

// 채널 구독 취소 시 나오는 모달창 끄기
document.querySelector(".meeting-cancel-btn").addEventListener("click", (e) => {
    let subModal = document.querySelector(".subcribe-wrap-cancel");
    subModal.style.display = "none";
});

// 슬라이드 배너 (이미 이미지가 존재 하기 때문에 별도로 추가되는 부분은 없음)

const slideContainer = document.querySelector(".swiper-wrap");
const pageCount = document.querySelector(".swiper-page-count");
let count = 0;
// 페이지 생성 시 count 가 초기 상태로 1로 존재 해야하기 때문에 사용
pageCount.innerText = `${count + 1}/2`;

const autoSlide = () => {
    slideContainer.style.transition = "transform 0.4s";
    count++;
    if (count == 2) {
        slideContainer.style.transform = "translate(-" + 358 * (count + 1) + "px)";
        setTimeout(() => {
            slideContainer.style.transition = "transform 0s";
            slideContainer.style.transform = "translate(-358px)";
        }, 500);
        count = 0;
        pageCount.innerText = `${count + 1}/2`;
    } else {
        pageCount.innerText = `${count + 1}/2`;
        slideContainer.style.transform = "translate(-" + 358 * (count + 1) + "px)";
    }
};

slideContainer.style.transform = "translate(-358px)";
let inter = setInterval(autoSlide, 4000);

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// 배너 + 선택 시 모달 생성
let plusBtn = document.querySelector(".swiper-more-button");
plusBtn.addEventListener("click", (e) => {
    let thumBanner = document.querySelector(".banner-plus-bg");
    thumBanner.style.display = "block";
});

// 배너 x 선택 시 모달 닫기
let exitBtn = document.querySelector(".absolute-exit-icon");
exitBtn.addEventListener("click", (e) => {
    let thumBanner = document.querySelector(".banner-plus-bg");
    thumBanner.style.display = "none";
});

//filter-container-hidden  그 필터 레이아웃
//.full-wrap 전체 background 이미지
//hidden-right-btn 필터 버튼
// hidden-flex-center 필터 닫는 버튼

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// 여는 방식
let filterButton = document.querySelector(".hidden-right-btn");
let filterContainer = document.querySelector(".filter-container-hidden");
let searchFilterButton = document.querySelector(".content-hidden-fiexd");
let mainBodyContainer = document.querySelector(".main-screen");

filterButton.addEventListener("click", (e) => {
    // 모달의 경우 가장 바깥쪽
    filterContainer.style.animation = "slideIn 0.3s ease-in-out";
    setTimeout(() => {
        mainBodyContainer.style.display = "none";
        // 애니메이션이 끝난 후 display 속성 변경
        filterContainer.style.display = "block";
    }, 300);
});

// 닫는 방식

searchFilterButton.addEventListener("click", (e) => {
    filterContainer.style.animation = "slideOut 0.3s ease-in-out";
    setTimeout(() => {
        filterContainer.removeAttribute("style");
        mainBodyContainer.style.display = "block";
        filterContainer.style.display = "none";
    }, 300);
});