function applyStyles(inputElement, redBoxElement) {
    if (!inputElement.value.trim()) {
        redBoxElement.classList.remove("hidden");
        inputElement.style.border = "2px solid  #CE201B";
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

// 축제 제목
var inputElementTitleba = document.querySelector(".admin-post-modal-pay-input");
var redBoxElementTitleba = document.getElementById("red-pay");
inputElementTitleba.addEventListener("input", function () {
    applyStyles(inputElementTitleba, redBoxElementTitleba);
});
inputElementTitleba.addEventListener("blur", function () {
    applyStyles(inputElementTitleba, redBoxElementTitleba);
});

// 축제 제목
var inputElementTitlebb = document.querySelector(
    ".admin-post-modal-adress-input"
);
var redBoxElementTitlebb = document.getElementById("red-adress");
inputElementTitlebb.addEventListener("input", function () {
    applyStyles(inputElementTitlebb, redBoxElementTitlebb);
});
inputElementTitlebb.addEventListener("blur", function () {
    applyStyles(inputElementTitlebb, redBoxElementTitlebb);
});

// 축제 제목
var inputElementTitlebc = document.querySelector(
    ".admin-post-modal-adressadd-input"
);
var redBoxElementTitlebc = document.getElementById("red-adressadd");
inputElementTitlebc.addEventListener("input", function () {
    applyStyles(inputElementTitlebc, redBoxElementTitlebc);
});
inputElementTitlebc.addEventListener("blur", function () {
    applyStyles(inputElementTitlebc, redBoxElementTitlebc);
});

// 축제 제목
var inputElementTitlebd = document.querySelector(
    ".admin-post-modal-place-input"
);
var redBoxElementTitlebd = document.getElementById("red-place");
inputElementTitlebd.addEventListener("input", function () {
    applyStyles(inputElementTitlebd, redBoxElementTitlebd);
});
inputElementTitlebd.addEventListener("blur", function () {
    applyStyles(inputElementTitlebd, redBoxElementTitlebd);
});

// 축제 제목
var inputElementTitlebe = document.querySelector(
    ".admin-post-modal-sttime-input"
);
var redBoxElementTitlebe = document.getElementById("red-sttime");
inputElementTitlebe.addEventListener("input", function () {
    applyStyles(inputElementTitlebe, redBoxElementTitlebe);
});
inputElementTitlebe.addEventListener("blur", function () {
    applyStyles(inputElementTitlebe, redBoxElementTitlebe);
});

// 축제 제목
var inputElementTitlebf = document.querySelector(
    ".admin-post-modal-endtime-input"
);
var redBoxElementTitlebf = document.getElementById("red-endtime");
inputElementTitlebf.addEventListener("input", function () {
    applyStyles(inputElementTitlebf, redBoxElementTitlebf);
});
inputElementTitlebf.addEventListener("blur", function () {
    applyStyles(inputElementTitlebf, redBoxElementTitlebf);
});

// 축제 제목
var inputElementTitlebg = document.querySelector(
    ".admin-post-modal-time-input"
);
var redBoxElementTitlebg = document.getElementById("red-time");
inputElementTitlebg.addEventListener("input", function () {
    applyStyles(inputElementTitlebg, redBoxElementTitlebg);
});
inputElementTitlebg.addEventListener("blur", function () {
    applyStyles(inputElementTitlebg, redBoxElementTitlebg);
});

// 축제 제목
var inputElementTitlebh = document.querySelector(
    ".admin-post-modal-opne-input"
);
var redBoxElementTitlebh = document.getElementById("red-opne");
inputElementTitlebh.addEventListener("input", function () {
    applyStyles(inputElementTitlebh, redBoxElementTitlebh);
});
inputElementTitlebh.addEventListener("blur", function () {
    applyStyles(inputElementTitlebh, redBoxElementTitlebh);
});

// 축제 제목
var inputElementTitlebi = document.querySelector(
    ".admin-post-modal-phone-input"
);
var redBoxElementTitlebi = document.getElementById("red-phone");
inputElementTitlebi.addEventListener("input", function () {
    applyStyles(inputElementTitlebi, redBoxElementTitlebi);
});
inputElementTitlebi.addEventListener("blur", function () {
    applyStyles(inputElementTitlebi, redBoxElementTitlebi);
});

// 축제 제목
var inputElementTitlebj = document.querySelector(
    ".admin-post-modal-give-input"
);
var redBoxElementTitlebj = document.getElementById("red-give");
inputElementTitlebj.addEventListener("input", function () {
    applyStyles(inputElementTitlebj, redBoxElementTitlebj);
});
inputElementTitlebj.addEventListener("blur", function () {
    applyStyles(inputElementTitlebj, redBoxElementTitlebj);
});

// 축제 제목
var inputElementTitlebk = document.querySelector(".admin-post-modal-url-input");
var redBoxElementTitlebk = document.getElementById("red-url");
inputElementTitlebk.addEventListener("input", function () {
    applyStyles(inputElementTitlebk, redBoxElementTitlebk);
});
inputElementTitlebk.addEventListener("blur", function () {
    applyStyles(inputElementTitlebk, redBoxElementTitlebk);
});
