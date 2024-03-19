// 버튼 클릭 시 상단으로 이동하는 js
const historyBtn = document.querySelector(".more-btn");
const cropLine = document.querySelector(".histroy-crop-line");
const contentBox = document.querySelector(".history-content-box");

historyBtn.addEventListener("click", (e) => {
    cropLine.style.borderTop = "none";
    historyBtn.style.display = "none";
    contentBox.style.height = "max-content";
});

const moveHistory = document.querySelector(".history-text");
const moveHistorySection = document.querySelector(".section-history");

moveHistory.addEventListener("click", function () {
    window.scrollTo({
        top: moveHistorySection.offsetTop,
        behavior: "smooth",
    });
});

const moveTeam = document.querySelector(".teams-text");
const moveTeamSection = document.querySelector(".section-teams");

moveTeam.addEventListener("click", function () {
    window.scrollTo({
        top: moveTeamSection.offsetTop,
        behavior: "smooth",
    });
});

const moveNews = document.querySelector(".news-text");
const moveNewsSection = document.querySelector(".section-news");

moveNews.addEventListener("click", function () {
    window.scrollTo({
        top: moveNewsSection.offsetTop,
        behavior: "smooth",
    });
});

const moveContact = document.querySelector(".contact-us-text");
const moveContactSection = document.querySelector(".section-contact-us");

moveContact.addEventListener("click", function () {
    window.scrollTo({
        top: moveContactSection.offsetTop,
        behavior: "smooth",
    });
});

// 모달 팝업 open / close
const memberProfileClick = document.querySelectorAll(".member");
const memberPop = document.querySelector(".teams-member-pop");
const memberPopBk = document.querySelector(".member-pop-background");
const memberPopIcon = document.querySelector(".pop-member-btn-close");
memberProfileClick.forEach((members) => {
    members.addEventListener("click", (e) => {
        memberPop.style.display = "flex";
        memberPop.style.alignItems = "center";
        memberPop.style.justifyContent = "center";
    });
});

memberPopBk.addEventListener("click", (e) => {
    memberPop.style.display = "none";
});

memberPopIcon.addEventListener("click", (e) => {
    memberPop.style.display = "none";
});

// 올라가기 아이콘 수정
const goTopIcon = document.querySelector(".scroll-to-top");

window.addEventListener("scroll", () => {
    if (window.scrollY <= 350) {
        goTopIcon.classList.add("hidden");
    } else {
        goTopIcon.classList.remove("hidden");
    }
});

goTopIcon.addEventListener("click", () => {
    window.scrollTo({
        top: 0,
        behavior: "smooth",
    });
});

// 우측 상단 아이콘 선택 시 modal 팝업 열기 및 클릭했을 때 눌렀을 때 닫기
// (animate 효과가 들어갑니다.)

let rightButton = document.querySelector(".menu-button-open");
let menuModal = document.querySelector(".menu-box");

rightButton.addEventListener("click", () => {
    fadeIn(menuModal);
});

let rightModalClose = document.querySelector(".menu-button-close");
menuModal.addEventListener("click", () => {
    fadeOut(menuModal);
});

// Function to fade in the menu modal
function fadeIn(element) {
    element.style.opacity = 0;
    element.style.display = "flex";

    let opacity = 0;

    let fadeInInterval = setInterval(() => {
        opacity += 0.05;
        element.style.opacity = opacity;
        if (opacity >= 1) {
            clearInterval(fadeInInterval);
        }
    }, 10);
}

// Function to fade out the menu modal
function fadeOut(element) {
    let opacity = 1;

    let fadeOutInterval = setInterval(() => {
        opacity -= 0.05;
        element.style.opacity = opacity;
        if (opacity <= 0) {
            clearInterval(fadeOutInterval);
            element.style.display = "none";
        }
    }, 10);
}

// 스크롤 위치 별 색상 변경
// 텍스트 색상 흰색 변경
let defaultHistoryText = document.querySelector(".history-text");
let teamsText = document.querySelector(".teams-text");
let newsText = document.querySelector(".news-text");
let contactUsText = document.querySelector(".contact-us-text");
//백그라운드 색깔 변경
let historyBackground = document.querySelector(".menu-item-history");
let teamsBackground = document.querySelector(".menu-item-teams");
let newsBackground = document.querySelector(".menu-item-news");
let contactUsBackground = document.querySelector(".menu-item-contact-us");

document.addEventListener("scroll", () => {
    let dexcriptionText = document.querySelector(".description-text");
    let teamContainerBox = document.querySelector(".teams-container-inner");
    let newsContainerBox = document.querySelector(".news-container");
    let contactUsContainer = document.querySelector(".news-more-btn");

    if (window.scrollY == 0) {
        defaultHistoryText.style.color = "rgb(93, 63, 191)";
        teamsText.style.color = "rgb(93, 63, 191)";
        newsText.style.color = "rgb(93, 63, 191)";
        contactUsText.style.color = "rgb(93, 63, 191)";
        historyBackground.style.backgroundColor = "";
        teamsBackground.style.backgroundColor = "";
        newsBackground.style.backgroundColor = "";
        contactUsBackground.style.backgroundColor = "";
    }
    if (window.scrollY > contactUsContainer.offsetTop) {
        defaultHistoryText.style.color = "rgb(93, 63, 191)";
        teamsText.style.color = "rgb(93, 63, 191)";
        newsText.style.color = "rgb(93, 63, 191)";
        contactUsText.style.color = "#fff";
        historyBackground.style.backgroundColor = "";
        teamsBackground.style.backgroundColor = "";
        newsBackground.style.backgroundColor = "";
        contactUsBackground.style.backgroundColor = "rgb(93, 63, 191)";
    }
    if (window.scrollY > newsContainerBox.offsetTop && window.scrollY < contactUsContainer.offsetTop) {
        defaultHistoryText.style.color = "rgb(93, 63, 191)";
        teamsText.style.color = "rgb(93, 63, 191)";
        newsText.style.color = "#fff";
        contactUsText.style.color = "rgb(93, 63, 191)";
        historyBackground.style.backgroundColor = "";
        teamsBackground.style.backgroundColor = "";
        newsBackground.style.backgroundColor = "rgb(93, 63, 191)";
        contactUsBackground.style.backgroundColor = "";
    }
    if (window.scrollY > teamContainerBox.offsetTop && window.scrollY < newsContainerBox.offsetTop) {
        defaultHistoryText.style.color = "rgb(93, 63, 191)";
        teamsText.style.color = "#fff";
        newsText.style.color = "rgb(93, 63, 191)";
        contactUsText.style.color = "rgb(93, 63, 191)";
        historyBackground.style.backgroundColor = "";
        teamsBackground.style.backgroundColor = "rgb(93, 63, 191)";
        newsBackground.style.backgroundColor = "";
        contactUsBackground.style.backgroundColor = "";
    }
    if (window.scrollY > dexcriptionText.offsetTop && window.scrollY < teamContainerBox.offsetTop) {
        defaultHistoryText.style.color = "#fff";
        teamsText.style.color = "rgb(93, 63, 191)";
        newsText.style.color = "rgb(93, 63, 191)";
        contactUsText.style.color = "rgb(93, 63, 191)";
        historyBackground.style.backgroundColor = "rgb(93, 63, 191)";
        teamsBackground.style.backgroundColor = "";
        newsBackground.style.backgroundColor = "";
        contactUsBackground.style.backgroundColor = "";
    }
});
