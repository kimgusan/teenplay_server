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

// 축제 제목
var inputElementTitlea = document.querySelector(
    ".admin-post-modal-title-input"
);
var redBoxElementTitlea = document.getElementById("red-title");
inputElementTitlea.addEventListener("input", function () {
    applyStyles(inputElementTitlea, redBoxElementTitlea);
});
inputElementTitlea.addEventListener("blur", function () {
    applyStyles(inputElementTitlea, redBoxElementTitlea);
});

// 축제 제목
var inputElementTitleb = document.querySelector(
    ".admin-post-modal-content-input"
);
var redBoxElementTitleb = document.getElementById("red-content");
inputElementTitleb.addEventListener("input", function () {
    applyStyles(inputElementTitleb, redBoxElementTitleb);
});
inputElementTitleb.addEventListener("blur", function () {
    applyStyles(inputElementTitleb, redBoxElementTitleb);
});
