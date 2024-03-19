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

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// 스크롤 위치 별 history team, news, contact us 항목
// 스크롤 위치 별 보라색 표기
const historyBg = document.querySelector(".menu-item-history");
const historyText = document.querySelector(".history-text");
const teamsBg = document.querySelector(".menu-item-teams");
const teamsText = document.querySelector(".teams-text");
const newsBg = document.querySelector(".menu-item-news");
const newsText = document.querySelector(".news-text");
const contactUsBg = document.querySelector(".menu-item-contact-us");
const contactUsText = document.querySelector(".contact-us-text");
const historyBtn2 = document.querySelector(".more-btn");
const hisStyle = getComputedStyle(historyBtn2);

window.addEventListener("scroll", function () {
    let sectionTeamsDiv = document.querySelector(".section-teams");
    let sectionNewsDiv = document.querySelector(".section-news");
    let sectionNewsMoreBtnBox = document.querySelector(".news-more-btn-box");

    if (window.scrollY <= 360) {
        historyBg.style.backgroundColor = "#fff";
        historyText.style.color = "#ce201b";
        teamsBg.style.backgroundColor = "#fff";
        teamsText.style.color = "#ce201b";
        newsBg.style.backgroundColor = "#fff";
        newsText.style.color = "#ce201b";
        contactUsBg.style.backgroundColor = "#fff";
        contactUsText.style.color = "#ce201b";
    }
    if (window.scrollY > 360 && window.scrollY <= sectionTeamsDiv.offsetTop) {
        historyBg.style.backgroundColor = "#ce201b";
        historyText.style.color = "white";
        teamsBg.style.backgroundColor = "#fff";
        teamsText.style.color = "#ce201b";
        newsBg.style.backgroundColor = "#fff";
        newsText.style.color = "#ce201b";
        contactUsBg.style.backgroundColor = "#fff";
        contactUsText.style.color = "#ce201b";
    }
    if (window.scrollY >= sectionTeamsDiv.offsetTop - 100 && window.scrollY < sectionNewsDiv.offsetTop) {
        teamsBg.style.backgroundColor = "#CE201B";
        teamsText.style.color = "#fff";
        historyBg.style.backgroundColor = "#fff";
        historyText.style.color = "#ce201b";
        newsBg.style.backgroundColor = "#fff";
        newsText.style.color = "#ce201b";
        contactUsBg.style.backgroundColor = "#fff";
        contactUsText.style.color = "#ce201b";
    }
    if (window.scrollY >= sectionNewsDiv.offsetTop - 100 && window.scrollY < sectionNewsMoreBtnBox.offsetTop) {
        newsBg.style.backgroundColor = "#CE201B";
        newsText.style.color = "#fff";
        historyBg.style.backgroundColor = "#fff";
        historyText.style.color = "#ce201b";
        teamsBg.style.backgroundColor = "#fff";
        teamsText.style.color = "#ce201b";
        contactUsBg.style.backgroundColor = "#fff";
        contactUsText.style.color = "#ce201b";
    }
    if (window.scrollY > sectionNewsMoreBtnBox.offsetTop - 100) {
        contactUsBg.style.backgroundColor = "#CE201B";
        contactUsText.style.color = "#fff";
        historyBg.style.backgroundColor = "#fff";
        historyText.style.color = "#ce201b";
        teamsBg.style.backgroundColor = "#fff";
        teamsText.style.color = "#ce201b";
        newsBg.style.backgroundColor = "#fff";
        newsText.style.color = "#ce201b";
    }
});

// 모달 팝업 open / close (팀원 클릭 시 나타나는 모달)
const memberProfileClick = document.querySelectorAll(".member");
const memberPop = document.querySelector(".teams-member-pop");
const memberPopBk = document.querySelector(".member-pop-background");
const memberPopIcon = document.querySelector(".pop-member-btn-close");
const memberPopupImage = document.querySelector(".pop-member-image-picture");
const memberImages = document.querySelectorAll(".member-image-picture")
const memberNames = document.querySelectorAll(".member-name");
const memberPositions = document.querySelectorAll(".member-appointment");
const memberPopName = document.querySelector(".pop-member-name");
const memberPopPosition = document.querySelector(".pop-member-appointment");
memberProfileClick.forEach((members, i) => {
    members.addEventListener("click", (e) => {
        memberPop.style.display = "flex";
        memberPop.style.alignItems = "center";
        memberPop.style.justifyContent = "center";
        memberPopupImage.style.backgroundImage = memberImages[i].style.backgroundImage;
        memberPopName.innerText = memberNames[i].innerText;
        memberPopPosition.innerText = memberPositions[i].innerText;
    });
});

memberPopBk.addEventListener("click", (e) => {
    memberPop.style.display = "none";
});

memberPopIcon.addEventListener("click", (e) => {
    memberPop.style.display = "none";
});

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// 올라가기 아이콘 수정 (최상단 올라가기)
const goTopIcon = document.querySelector(".scroll-to-top");

window.addEventListener("scroll", () => {
    if (window.scrollY <= 350) {
        goTopIcon.classList.add("hidden-up");
    } else {
        goTopIcon.classList.remove("hidden-up");
    }
});

goTopIcon.addEventListener("click", () => {
    window.scrollTo({
        top: 0,
        behavior: "smooth",
    });
});

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// 맴버 more button
let memberMoreButton = document.querySelector(".member-more-btn-box");
memberMoreButton.addEventListener("click", (e) => {
    let memberDisplay = document.querySelectorAll(".member-display.hidden");
    memberDisplay.forEach((member) => {
        member.classList.remove("hidden");
        memberMoreButton.classList.add("hidden");
    });
});

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// // 카카오 맵
// var mapContainer = document.getElementById("map"), // 지도를 표시할 div
//     mapOption = {
//         center: new kakao.maps.LatLng(33.450701, 126.570667), // 지도의 중심좌표
//         level: 3, // 지도의 확대 레벨
//     };
//
// // 지도를 표시할 div와  지도 옵션으로  지도를 생성합니다
// var map = new kakao.maps.Map(mapContainer, mapOption);


// 공지사항 목록 불러오기


// 공지사항 더보기버튼
const noticeMoreBtn = document.querySelector(".news-more-btn-box")

let page = 1;
const getNoticeList = (callback) => {
    fetch(`/company/notice/api/${page}`)
        .then((response) => response.json())
        .then((notice_info) => {
            if (callback) {
                callback(notice_info);
            }
        });
};

// 각 공지사항 클릭 시 링크 연결은 공지사항 페이지 작업 끝나고 하도록 하겠습니다.
const showNoticeList = (notice_info) => {
    if (!notice_info.hasNext){
        noticeMoreBtn.style.display = "none";
    }
    let notices = notice_info.notices;
    const newsBox = document.querySelector(".news-box")
    notices.forEach((notice) => {
        newsBox.innerHTML += `
            <a class="news-list" href="" target="_blank">
                <div class="news-head">
                    <div class="news-date">${notice.created_date.slice(0, 10)}</div>
                </div>
                <div class="news-body">
                    <div class="news-subject">${notice.notice_title}</div>
                    <div class="news-description">
                        ${notice.notice_content}
                    </div>
                </div>
            </a>
        `
    })
    updateNoticeCount(notice_info.total);
}

getNoticeList(showNoticeList);

// 버튼 안의 숫자 바꾸기
const noticeMoreBtnCount = document.querySelector(".news-more-btn")
const updateNoticeCount = (total) => {
    const nowShowingCount = document.querySelectorAll(".news-list").length;
    noticeMoreBtnCount.innerText = `More Notices.. (${nowShowingCount}/${total})`
}
noticeMoreBtn.addEventListener("click", (e) => {
    page++;
    getNoticeList(showNoticeList);
})