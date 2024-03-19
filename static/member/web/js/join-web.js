// 회원가입 버튼 쿼리
const joinButton = document.querySelector(".join-wrap");
// 버튼 클릭 시 한번 더 검사하기 위해
let flag = false;

// form 태그 가져오기
const joinForm = document.querySelector("form.input-all-wrap");

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

// 전화번호 입력 시 이벤트 발생
// 전화번호 글자수 검사
let phoneValue = false;
// 전화번호 정규식(숫자만)
let phoneRegex = /^[0-9]*$/;
// 전화번호 입력 쿼리
const phoneInput = document.querySelector(".phone-input")
phoneInput.addEventListener("keyup", (e) => {
    if (e.target.value){
        if (!phoneRegex.test(e.target.value) || e.target.value.length < 10){
            phoneInput.classList.add("color")
            phoneValue = false;
        } else {
            phoneInput.classList.remove("color");
            phoneValue = true;
        }
    }
    allCheck();
})


// 약관동의
NodeList.prototype.filter = Array.prototype.filter;

let essentialValue = false;
// 필수 약관동의 체크 박스 쿼리(다수)
const essentialAgreement = document.querySelectorAll(".agreement-button.essential");

// 필수 약관동의 체크 박스가 3개가 체크가 안되면 오류
essentialAgreement.forEach((agreement) => {
    agreement.addEventListener("click", () => {
        essentialValue = essentialAgreement.filter((agreement) => agreement.checked).length === 3;
        nameValue = nameInput.value.length >= 2;
        allCheck();
    });
});

// 전체 검사
// 모든 값에 오류가 없으면 회원가입 버튼 활성화
// 오류가 있으면 비활성화

function allCheck() {
    if (nameValue && phoneValue && essentialValue) {
        joinButton.classList.remove("disabled");
        flag = true;
        return;
    }
    joinButton.classList.add("disabled");
    flag = false;
}

joinButton.addEventListener("click", (e) => {
    if (flag) {
        joinForm.submit();
    }
})