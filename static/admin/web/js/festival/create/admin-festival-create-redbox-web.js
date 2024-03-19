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

// 이용요금
var inputElementPayment = document.querySelector(
    ".main-festival-plan-detail-pay-input"
);
var redBoxElementPayment = document.getElementById("red-pay");
inputElementPayment.addEventListener("input", function () {
    applyStyles(inputElementPayment, redBoxElementPayment);
});
inputElementPayment.addEventListener("blur", function () {
    applyStyles(inputElementPayment, redBoxElementPayment);
});

// 축제 제목
var inputElementTitle = document.querySelector(
    ".main-festival-info-detail-input"
);
var redBoxElementTitle = document.getElementById("red-title");
inputElementTitle.addEventListener("input", function () {
    applyStyles(inputElementTitle, redBoxElementTitle);
});
inputElementTitle.addEventListener("blur", function () {
    applyStyles(inputElementTitle, redBoxElementTitle);
});

// 축제 내용
var inputElementContent = document.querySelector(
    ".main-festival-info-detail-textarea"
);
var redBoxElementContent = document.getElementById("red-content");
inputElementContent.addEventListener("input", function () {
    applyStyles(inputElementContent, redBoxElementContent);
});
inputElementContent.addEventListener("blur", function () {
    applyStyles(inputElementContent, redBoxElementContent);
});

// 주소
var inputElementAddress = document.querySelector(
    ".main-festival-plan-detail-adress-input"
);
var redBoxElementAddress = document.getElementById("red-address");
inputElementAddress.addEventListener("input", function () {
    applyStyles(inputElementAddress, redBoxElementAddress);
});
inputElementAddress.addEventListener("blur", function () {
    applyStyles(inputElementAddress, redBoxElementAddress);
});

// 상세
var inputElementAddressdetail = document.querySelector(
    ".main-festival-plan-detail-adressadd-input"
);
var redBoxElementAddressdetail = document.getElementById("red-addressdetail");
inputElementAddressdetail.addEventListener("input", function () {
    applyStyles(inputElementAddressdetail, redBoxElementAddressdetail);
});
inputElementAddressdetail.addEventListener("blur", function () {
    applyStyles(inputElementAddressdetail, redBoxElementAddressdetail);
});

// 행사장소
var inputElementPlace = document.querySelector(
    ".main-festival-plan-detail-place-input"
);
var redBoxElementPlace = document.getElementById("red-place");
inputElementPlace.addEventListener("input", function () {
    applyStyles(inputElementPlace, redBoxElementPlace);
});
inputElementPlace.addEventListener("blur", function () {
    applyStyles(inputElementPlace, redBoxElementPlace);
});

// 시작일
var inputElementStart = document.querySelector(
    ".main-festival-plan-detail-startdate-input"
);
var redBoxElementStart = document.getElementById("red-start");
inputElementStart.addEventListener("input", function () {
    applyStyles(inputElementStart, redBoxElementStart);
});
inputElementStart.addEventListener("blur", function () {
    applyStyles(inputElementStart, redBoxElementStart);
});

// 종료일
var inputElementLast = document.querySelector(
    ".main-festival-plan-detail-lastdate-input"
);
var redBoxElementLast = document.getElementById("red-last");
inputElementLast.addEventListener("input", function () {
    applyStyles(inputElementLast, redBoxElementLast);
});
inputElementLast.addEventListener("blur", function () {
    applyStyles(inputElementLast, redBoxElementLast);
});

// 종료일
var inputElementTime = document.querySelector(
    ".main-festival-plan-detail-time-input"
);
var redBoxElementTime = document.getElementById("red-time");
inputElementTime.addEventListener("input", function () {
    applyStyles(inputElementTime, redBoxElementTime);
});
inputElementTime.addEventListener("blur", function () {
    applyStyles(inputElementTime, redBoxElementTime);
});

// 주최자
var inputElementInfo = document.querySelector(
    ".main-festival-admin-detail-info-input"
);
var redBoxElementInfo = document.getElementById("red-info");
inputElementInfo.addEventListener("input", function () {
    applyStyles(inputElementInfo, redBoxElementInfo);
});
inputElementInfo.addEventListener("blur", function () {
    applyStyles(inputElementInfo, redBoxElementInfo);
});

// 주최자 번호
var inputElementPhone = document.querySelector(
    ".main-festival-admin-detail-phone-input"
);
var redBoxElementPhone = document.getElementById("red-phone");
inputElementPhone.addEventListener("input", function () {
    applyStyles(inputElementPhone, redBoxElementPhone);
});
inputElementPhone.addEventListener("blur", function () {
    applyStyles(inputElementPhone, redBoxElementPhone);
});

// 제공자
var inputElementGive = document.querySelector(
    ".main-festival-give-detail-info-input"
);
var redBoxElementGive = document.getElementById("red-give");
inputElementGive.addEventListener("input", function () {
    applyStyles(inputElementGive, redBoxElementGive);
});
inputElementGive.addEventListener("blur", function () {
    applyStyles(inputElementGive, redBoxElementGive);
});

// url
var inputElementUrl = document.querySelector(
    ".main-festival-give-detail-phone-input"
);
var redBoxElementUrl = document.getElementById("red-url");
inputElementUrl.addEventListener("input", function () {
    applyStyles(inputElementUrl, redBoxElementUrl);
});
inputElementUrl.addEventListener("blur", function () {
    applyStyles(inputElementUrl, redBoxElementUrl);
});
