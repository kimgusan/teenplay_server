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
var inputElementTitle = document.querySelector(".admin-post-modal-title-input");
var redBoxElementTitle = document.getElementById("red-title");
inputElementTitle.addEventListener("input", function () {
    applyStyles(inputElementTitle, redBoxElementTitle);
});
inputElementTitle.addEventListener("blur", function () {
    applyStyles(inputElementTitle, redBoxElementTitle);
});
