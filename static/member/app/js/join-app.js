// 회원가입 버튼 쿼리
const joinButton = document.querySelector(".join-wrap");

// 이메일 형식 검사
let emailValue = false;
// 이메일 입력 쿼리
const emailInput = document.querySelector(".email-input");
// 이메일 오류 메시지 쿼리
const emailLength = document.querySelector(".email-length-error");
// 이메일 형식 검사 
const emailRegex = /^[a-zA-Z0-9+-\_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$/;

// 이메일 입력 시 이벤트 발생
// 값이 형식에 맞지 않으면 오류 메시지 및 붉은색 표시
emailInput.addEventListener("keyup", (e) => {
    if (!emailRegex.test(e.target.value)) {
        emailLength.classList.remove("hidden");
        emailInput.classList.add("color");
        emailValue = false;
    } else {
        emailLength.classList.add("hidden");
        emailInput.classList.remove("color");
        emailValue = true;
    }
    allCheck();
});

// 이름 글자수 검사
let nameValue = false;
// 이름 입력 쿼리
const nameInput = document.querySelector(".name-input");
// 이름 입력 오류 메시지 쿼리
const nameLengthError = document.querySelector(".name-length-error");

// 이름 입력 시 이벤트 발생
// 값이 있는데, 값이 2 미만이면 에러
nameInput.addEventListener("keyup", (e) => {
    if (e.target.value) {
        if (e.target.value.length < 2) {
            nameLengthError.classList.remove("hidden");
            nameInput.classList.add("color");
            nameValue = false;
        } else {
            nameLengthError.classList.add("hidden");
            nameInput.classList.remove("color");
            nameValue = true;
        }
    } 
    allCheck();
});

// 비밀번호 글자수 검사
let passwordValue = false;
// 비밀번호 입력 쿼리
const passwordInput = document.querySelector(".password-input");
// 비밀번호 에러 메시지 쿼리
const passwordLengthError = document.querySelector(".password-length-error");

// 비밀번호 입력 시 이벤트 발생
// 값이 6보다 적으면 에러 메시지 활성화
passwordInput.addEventListener("keyup", (e) => {
    if (e.target.value) {
        if (e.target.value.length < 6) {
            passwordLengthError.classList.remove("hidden");
            passwordInput.classList.add("color");
            passwordValue = false;
        } else {
            passwordLengthError.classList.add("hidden");
            passwordInput.classList.remove("color");
            passwordValue = true;
        }
    }
    allCheck();
});

// 약관동의
NodeList.prototype.filter = Array.prototype.filter;

let essentialValue = false;
// 필수 약관동의 체크 박스 쿼리(다수)
const essentialAgreement = document.querySelectorAll(".agreement-button.essential");

// 필수 약관동의 체크 박스가 3개가 체크가 안되면 오류
essentialAgreement.forEach((agreement) => {
    agreement.addEventListener("click", () => {
        essentialValue = essentialAgreement.filter((agreement) => agreement.checked).length === 3;
        allCheck();
    });
});

// 전체 검사
// 모든 값에 오류가 없으면 회원가입 버튼 활성화
// 오류가 있으면 비활성화
function allCheck() {
    if (emailValue && nameValue && passwordValue && essentialValue) {
        joinButton.classList.remove("disabled");
        return;
    }
    joinButton.classList.add("disabled");
}
