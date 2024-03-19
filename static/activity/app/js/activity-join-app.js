// 동의합니다 radio 버튼 선택 체크
let surveyRadioTie = document.querySelector(".survey-radio-tie");
surveyRadioTie.addEventListener("click", (e) => {
    let radio = document.querySelector(".ev-radio-111");
    if (!radio.checked) {
        radio.checked = "true";
    }
});

// 동의하기 약관 checkbox 전체 버튼 및 개별 선택 체크
// let aggreementBetweenCenter = document.querySelector(".aggreement-between-center");
let aggreementAllChecks = document.querySelector(".aggreement-all-checks");
let aggreementAgeCheck = document.querySelector(".aggreement-age-check");
let aggreementPersonerCheck = document.querySelector(".aggreement-personerinfo-check");
let aggreementServiceCheck = document.querySelector(".aggreement-service-check");
let aggreementPrivacyCheck = document.querySelector(".aggreement-privacy-check");
let aggreementPayCheck = document.querySelector(".aggreement-pay-check");

let aggreementAnothers = document.querySelectorAll(".aggreement-between-center");

// 체크박스 경고문구 제거를 위한 변수 선언
let dangerText = document.querySelectorAll(".aggreement-text-danger");

aggreementAnothers.forEach((another, i) => {
    another.addEventListener("click", (e) => {
        if (i == 0) {
            if (aggreementAllChecks.checked) {
                aggreementAllChecks.checked = true;
                aggreementAgeCheck.checked = true;
                aggreementPersonerCheck.checked = true;
                aggreementServiceCheck.checked = true;
                aggreementPayCheck.checked = true;
                aggreementPrivacyCheck.checked = true;
                dangerText[i].classList.remove("none");
                dangerText[i + 1].classList.remove("none");
                dangerText[i + 2].classList.remove("none");
                dangerText[i + 3].classList.remove("none");
                dangerText[i + 4].classList.remove("none");
            } else {
                aggreementAllChecks.checked = false;
                aggreementAgeCheck.checked = false;
                aggreementPersonerCheck.checked = false;
                aggreementServiceCheck.checked = false;
                aggreementPayCheck.checked = false;
                aggreementPrivacyCheck.checked = false;
            }
        }
        if (i == 1) {
            if (aggreementAgeCheck.checked) {
                aggreementAgeCheck.checked = true;
                dangerText[i - 1].classList.remove("none");
            } else {
                aggreementAgeCheck.checked = false;
            }
        }
        if (i == 2) {
            if (aggreementPersonerCheck.checked) {
                aggreementPersonerCheck.checked = true;
                dangerText[i - 1].classList.remove("none");
            } else {
                aggreementPersonerCheck.checked = false;
            }
        }
        if (i == 3) {
            if (aggreementServiceCheck.checked) {
                aggreementServiceCheck.checked = true;
                dangerText[i - 1].classList.remove("none");
            } else {
                aggreementServiceCheck.checked = false;
            }
        }
        if (i == 4) {
            if (aggreementPrivacyCheck.checked) {
                aggreementPrivacyCheck.checked = true;
                dangerText[i - 1].classList.remove("none");
            } else {
                aggreementPrivacyCheck.checked = false;
            }
        }
        if (i == 5) {
            if (aggreementPayCheck.checked) {
                aggreementPayCheck.checked = true;
                dangerText[i - 1].classList.remove("none");
            } else {
                aggreementPayCheck.checked = false;
            }
        }
    });
});

// 동의하기 약관 개별 선택 시 전체 동의하기 체크
aggreementAnothers.forEach((another, i) => {
    another.addEventListener("click", (e) => {
        if (!aggreementAllChecks.checked && aggreementAgeCheck.checked && aggreementPersonerCheck.checked && aggreementServiceCheck.checked && aggreementPrivacyCheck.checked && aggreementPayCheck.checked) {
            aggreementAllChecks.checked = true;
        } else if ((aggreementAllChecks.checked && !aggreementAgeCheck.checked) || !aggreementPersonerCheck.checked || !aggreementServiceCheck.checked || !aggreementPrivacyCheck.checked || !aggreementPayCheck.checked) {
            aggreementAllChecks.checked = false;
        }
    });
});

// 신청하기/ 결제하기 버튼 클릭 시 공란이 있으면 발생되는 이벤트
let button = document.querySelector(".no-pay-button");
let nameInput = document.querySelector(".input-name");
let phoneInput = document.querySelector(".input-address-phone");
let purposeInput = document.querySelector(".input-purpose");
let surveyRadioCheck = document.querySelector(".ev-radio-111");

let checkBox = document.querySelectorAll(".aggreement-items-box");

let inputDangerText = document.querySelectorAll(".aggreement-items-box input");

button.addEventListener("click", (e) => {
    if (nameInput.value === "") {
        let nameHidden = document.querySelector(".name-hidden-display");
        nameHidden.style.display = "flex";
        nameInput.classList.add("invalid-input");
        nameInput.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    if (!surveyRadioCheck.checked) {
        let radioCheckText = document.querySelector(".survey-important-text");
        radioCheckText.style.display = "flex";
        let surveyBox = document.querySelector(".survey-content-box");
        surveyBox.classList.add("invalid-input");
        if (nameInput.value != "") {
            surveyRadioCheck.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }
    inputDangerText.forEach((inputText, i) => {
        console.log(inputText.checked);
        if (!inputText.checked) {
            dangerText[i].classList.add("none");
        }
    });
});

// 이름을 입력하면 경고창 해제
nameInput.addEventListener("keyup", (e) => {
    if (nameInput.value != "") {
        let nameHidden = document.querySelector(".name-hidden-display");
        nameHidden.style.display = "none";
        nameInput.classList.remove("invalid-input");
    }
});

// radio 버튼 선택하면 경고창 해제
surveyRadioTie.addEventListener("click", () => {
    if (surveyRadioCheck.checked) {
        let radioCheckText = document.querySelector(".survey-important-text");
        radioCheckText.style.display = "none";
        let surveyBox = document.querySelector(".survey-content-box");
        surveyBox.classList.remove("invalid-input");
    }
});

// input 요소에 이메일 넣기
const inputEmail = document.querySelector(".input-email");
inputEmail.value = "kimkyusam@hello.com";
