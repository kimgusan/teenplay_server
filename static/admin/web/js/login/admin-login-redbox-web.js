function applyStyles(inputElement, redBoxElement) {
    if (!inputElement.value.trim()) {
        redBoxElement.classList.remove("hidden");
        inputElement.style.border = "2px solid #CE201B";
        // placeholder 스타일 설정
        inputElement.style.paddingLeft = "18px";
        inputElement.style.paddingRight = "18px";
        inputElement.style.borderRadius = "8px";
    } else {
        redBoxElement.classList.add("hidden");
        inputElement.style.border = "";
        // placeholder 스타일 초기화
        inputElement.style.paddingLeft = "";
        inputElement.style.paddingRight = "";
        inputElement.style.borderRadius = "";
    }
}

// 입력 박스(아이디)
var inputElementTitle = document.querySelector(".main-login-id");
var redBoxElementTitle = document.getElementById("red-id");
inputElementTitle.addEventListener("input", function () {
    applyStyles(inputElementTitle, redBoxElementTitle);
});
inputElementTitle.addEventListener("blur", function () {
    applyStyles(inputElementTitle, redBoxElementTitle);
});

// 입력 박스(비밀번호)
var inputElementContent = document.querySelector(".main-login-pw");
var redBoxElementContent = document.getElementById("red-pw");
inputElementContent.addEventListener("input", function () {
    applyStyles(inputElementContent, redBoxElementContent);
});
inputElementContent.addEventListener("blur", function () {
    applyStyles(inputElementContent, redBoxElementContent);
});
