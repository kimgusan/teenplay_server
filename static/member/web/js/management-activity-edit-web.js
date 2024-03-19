const activityForm = document.querySelector("form[name=activity-create]")
const updateButton = document.getElementById("last")
updateButton.addEventListener("click", async ()=>{
    let summernoteContent = $('.presentation-size').summernote('code');
    let activityContent = document.createElement("input")
    activityContent.setAttribute("type", "hidden");
    activityContent.setAttribute("name", "activity-content")
    activityContent.setAttribute("value", JSON.stringify(summernoteContent))
    activityForm.appendChild(activityContent)
    await activityForm.submit()
})
//datepicker
$(function () {
    $(".datepicker").datepicker();
});

$(function () {
    //input을 datepicker로 선언
    $(".datepicker").datepicker({
        dateFormat: "yy-mm-dd", //달력 날짜 형태
        showOtherMonths: true, //빈 공간에 현재월의 앞뒤월의 날짜를 표시
        showMonthAfterYear: true, // 월- 년 순서가아닌 년도 - 월 순서
        changeYear: true, //option값 년 선택 가능
        changeMonth: true, //option값  월 선택 가능
        showOn: "both", //button:버튼을 표시하고,버튼을 눌러야만 달력 표시 ^ both:버튼을 표시하고,버튼을 누르거나 input을 클릭하면 달력 표시
        buttonImage: "http://jqueryui.com/resources/demos/datepicker/images/calendar.gif", //버튼 이미지 경로
        buttonImageOnly: true, //버튼 이미지만 깔끔하게 보이게함
        buttonText: "선택", //버튼 호버 텍스트
        yearSuffix: "년", //달력의 년도 부분 뒤 텍스트
        monthNamesShort: ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"], //달력의 월 부분 텍스트
        monthNames: ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"], //달력의 월 부분 Tooltip
        dayNamesMin: ["일", "월", "화", "수", "목", "금", "토"], //달력의 요일 텍스트
        dayNames: ["일요일", "월요일", "화요일", "수요일", "목요일", "금요일", "토요일"], //달력의 요일 Tooltip
        minDate: "-5Y", //최소 선택일자(-1D:하루전, -1M:한달전, -1Y:일년전)
        maxDate: "+5y", //최대 선택일자(+1D:하루후, -1M:한달후, -1Y:일년후)
    });
});

$(".timepicker").timepicker({
    timeFormat: "HH:mm",
    interval: 60,
    minTime: "00:00",
    maxTime: "23:00pm",
    startTime: "00:00",
    dynamic: false,
    dropdown: true,
    scrollbar: true,
});

// 클릭 시 위치가 아래쪽에 생겨서 변경하기 위한 js
// 클래스가 "datepicker"인 input 요소가 클릭되었을 때 js
jQuery(".datepicker").datepicker({
    beforeShow: function (input) {
        // 클릭 이벤트가 발생한 위치를 가져옵니다.
        var clickX = jQuery(input).offset().left;
        var clickY = jQuery(input).offset().top + jQuery(input).outerHeight();
        // datepicker의 위치를 클릭 이벤트가 발생한 위치로 설정합니다.
        setTimeout(function () {
            jQuery("#ui-datepicker-div").css({ left: clickX + "px", top: clickY - 70 + "px" });
        });
    },
});

// 클래스가 "timepicker"인 input 요소가 클릭되었을 때 js
jQuery(document).on("focus", ".timepicker", function () {
    // 클릭된 input 요소를 선택
    var input = jQuery(this);
    // input 요소의 위치와 값을 로그에 출력
    var clickX = jQuery(input).offset().left;
    var clickY = jQuery(input).offset().top + jQuery(input).outerHeight();
    setTimeout(function () {
        jQuery(".ui-timepicker-container").css({ left: clickX + "px", top: clickY });
    }, 0);
});

// 텍스트 에디터
let fileNames = [];

$(document).ready(function () {
    //썸머노트에 값넣기 (차후 값을 넣었을 때 저장하기 위한 코드)
    // $(".presentation-size").summernote("code", "입력된 텍스트를 넣으세요");

    //위와 같이 값을 먼저 넣어준 후 초기화를 시킨다. 그럼 아래와 같이 입력이 된다.
    //초기화
    $(".presentation-size").summernote({
        height: 400, // set editor height
        minHeight: null, // set minimum height of editor
        maxHeight: null, // set maximum height of editor
        focus: false,
        lang: "ko-KR", // 기본 메뉴언어 US->KR로 변경
        callbacks: {
            onImageUpload: async function (files) {
                // 개수 제한 안 두고 하겠습니다.
                // let fileImages = document.querySelectorAll("div.note-editor img");
                // if (fileImages) {
                //     if (fileImages.length === 2) {
                //         alert('2개 이하의 이미지만 첨부할 수 있습니다.');
                //         return;
                //     }
                // }
                const [file] = files;
                if (file.size >= 1024 * 1024 * 5) {
                    alert('5MB 이하의 이미지만 첨부할 수 있습니다.');
                    return;
                }
                // fileNames.push(file.name);
                // let fileInput = document.createElement("input");
                // fileInput.setAttribute("type", "file");
                // fileInput.setAttribute("style", "display: none;");
                // fileInput.setAttribute("name", "files");
                // let dataTransfer = new DataTransfer();
                // dataTransfer.items.add(file);
                // fileInput.files = dataTransfer.files;
                // let reader = new FileReader();
                // reader.readAsDataURL(file);
                // reader.addEventListener("load", (e) => {
                //     $(".presentation-size").summernote("insertImage", e.target.result);
                // })
                // activityForm.appendChild(fileInput);
                let formData = new FormData();
                formData.append('image', file);
                const response = await fetch(`/activity/images/api/`, {
                    method: "POST",
                    body: formData,
                    headers: {
                        'X-CSRFToken': csrfToken
                    }
                });
                const data = await response.json();
                const imageUrl = await data.image_path;
                const imageId = await data.image_id;
                $(".presentation-size").summernote('insertImage', imageUrl);
                let imageIdInput = document.createElement("input");
                imageIdInput.setAttribute("type", "hidden");
                imageIdInput.setAttribute("style", "display: none;");
                imageIdInput.setAttribute("name", "image-id");
                imageIdInput.setAttribute("value", imageId);
                activityForm.appendChild(imageIdInput);
            },
        }
    });
    $(".presentation-size").summernote('code', activityContent);
    // //저장버튼 클릭( 행사 게시 클릭 시 조건부로 만들어서 저장할 것)
    // $(document).on("click", "#saveBtn", function () {
    //     saveContent();
    // });
});

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  위에까지는 jquery 가져오기

// 행사 결제 클릭 시 입력되지 않은 부분 확인 후 붉은 색 문구 출력
// 버튼 클릭
let clickOpenButton = document.querySelector(".button-activity-open");

// 행사명
let errorMessagesBox = document.querySelector(".error-message-box");
let activityTitle = document.querySelector(".input-activity-title");

// 날짜
let errorMessagesBoxDate = document.querySelectorAll(".error-message-box-date");
let dateBoxAll = document.querySelectorAll(".datepicker");

// 시간
let errorMessagesBoxTime = document.querySelectorAll(".error-message-box-time");
let timeBoxAll = document.querySelectorAll(".timepicker");

// select
let errorMessagesBoxSelect = document.querySelector(".error-message-box-select");
let selectBox = document.querySelector(".subject-category");

// location

let errorMessagesBoxAddress = document.querySelector(".error-message-box-address");
let errorMessagesBoxLocation = document.querySelector(".error-message-box-location");

// let locationMapTitleInputText = document.querySelector(".location-map-title-input-text");
let locationNameInputText = document.querySelector(".location-name-input-text");
let locationContentInputText = document.querySelector(".location-content-input-text");


// 결제하기 버튼을 클릭했을 때 입력되지 않은 경고 문구 값 나오기
clickOpenButton.addEventListener("click", (e) => {
    if (activityTitle.value === "") {
        errorMessagesBox.style.display = "block";
        activityTitle.classList.add("border-color");
    }

    if (selectBox.value === "disabled") {
        errorMessagesBoxSelect.style.display = "block";
        selectBox.classList.add("border-color");
    }

    dateBoxAll.forEach((date, i) => {
        if (dateBoxAll[i].value === "") {
            errorMessagesBoxDate[i].style.display = "block";
            dateBoxAll[i].classList.add("border-color");
        }
    });

    timeBoxAll.forEach((time, i) => {
        if (timeBoxAll[i].value === "") {
            errorMessagesBoxTime[i].style.display = "block";
            timeBoxAll[i].classList.add("border-color");
        }
    });

    // if (locationSelect.value === "place" && locationMapTitleInputText.value === "" && locationNameInputText.value === "") {
    //     errorMessagesBoxAddress.style.display = "block";
    //     errorMessagesBoxLocation.style.display = "block";
    //     locationMapTitleInputText.classList.add("border-color");
    //     locationNameInputText.classList.add("border-color");
    // } else if (locationMapTitleInputText.value != "" && locationNameInputText.value === "") {
    //     errorMessagesBoxLocation.style.display = "block";
    //     locationNameInputText.classList.add("border-color");
    // }

    // if (locationSelect.value === "direct" && locationNameInputText.value === "") {
    //     errorMessagesBoxLocation.style.display = "block";
    //     locationNameInputText.classList.add("border-color");
    // }

    let dateBoxAllArray = [...document.querySelectorAll(".date-box")];
    let timeBoxAllArray = [...document.querySelectorAll(".date-box")];

    if (activityTitle.value != "" && selectBox.value != "disabled" && dateBoxAllArray.every((date) => date.value != "") && timeBoxAllArray.every((time) => time.value != "")) {
        pay();
    } else {
        window.scrollTo({
            top: 0,
            behavior: "smooth",
        });
    }
});

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// 경고창 해제
// 날짜 클릭 시 경고창 해제
dateBoxAll.forEach((date, i) => {
    date.addEventListener("focus", () => {
        errorMessagesBoxDate[i].style.display = "none";
        date.classList.remove("border-color");
    });
});

// 일자 클릭 시 경고창 해제
timeBoxAll.forEach((time, i) => {
    time.addEventListener("focus", () => {
        errorMessagesBoxTime[i].style.display = "none";
        time.classList.remove("border-color");
    });
});

// 유형 선택 시 경고창 해제
selectBox.addEventListener("change", () => {
    if (selectBox.value != "disabled") {
        errorMessagesBoxSelect.style.display = "none";
        selectBox.classList.remove("border-color");
    }
});

// 행사명 입력 시 경고창 해제
activityTitle.addEventListener("keyup", () => {
    errorMessagesBox.style.display = "none";
    activityTitle.classList.remove("border-color");
});

// 장소 입력 시 경고창 해제
locationNameInputText.addEventListener("keyup", () => {
    errorMessagesBoxLocation.style.display = "none";
    locationNameInputText.classList.remove("border-color");
});

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// // 장소 선택에 따른 지도/ 장소 input 보여주기
// let locationSelect = document.querySelector(".location-select-default");
// let locationMap = document.querySelector(".activity-map-col");
// let locationMapInput = document.querySelector(".input-location-map-title-col");
// let locationText = document.querySelector(".input-location-name-col");
// let locationTextDetail = document.querySelector(".input-location-content-col");
// locationSelect.addEventListener("change", () => {
//     if (locationSelect.value == "place") {
//         locationMapInput.style.display = "block";
//         locationMap.style.display = "block";
//         locationText.style.display = "block";
//         locationTextDetail.style.display = "block";
//     } else if (locationSelect.value == "direct") {
//         locationMapInput.style.display = "none";
//         locationMap.style.display = "none";
//         locationText.style.display = "block";
//         locationTextDetail.style.display = "block";
//     } else {
//         locationMapInput.style.display = "none";
//         locationMap.style.display = "none";
//         locationText.style.display = "none";
//         locationTextDetail.style.display = "none";
//     }
// });

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// 확장 배너 추가 시 선택하는 체크박스
let bannerCheck = document.querySelector(".back-thumnail-fill");
let expandAddBanner = document.querySelector(".thumnail-expand-add-box");
bannerCheck.addEventListener("click", () => {
    if (bannerCheck.checked) {
        expandAddBanner.style.display = "block";
    } else {
        expandAddBanner.style.display = "none";
    }
});
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// 화살표 클릭시 접기 펴기
let clickArrorwsUp = document.querySelectorAll(".material-symbols-outlined.mdi-up");
let clickArrorwsDown = document.querySelectorAll(".material-symbols-outlined.mdi-down");
// 접기
clickArrorwsUp.forEach((clickArrorw, i) => {
    clickArrorwsUp[i].addEventListener("click", (e) => {
        if (i == 0) {
            let normalCheckItems = document.querySelector(".normal-check-items");
            normalCheckItems.style.display = "none";
            clickArrorwsUp[i].style.display = "none";
            clickArrorwsDown[i].style.display = "block";
        } else if (i == 1) {
            let detailCheckItems = document.querySelector(".detail-all-box");
            detailCheckItems.style.display = "none";
            clickArrorwsUp[i].style.display = "none";
            clickArrorwsDown[i].style.display = "block";
        }
    });
});
// 펴기
clickArrorwsDown.forEach((clickArrorw, i) => {
    clickArrorwsDown[i].addEventListener("click", (e) => {
        if (i == 0) {
            let normalCheckItems = document.querySelector(".normal-check-items");
            normalCheckItems.style.display = "block";
            clickArrorwsUp[i].style.display = "block";
            clickArrorwsDown[i].style.display = "none";
        } else if (i == 1) {
            let detailCheckItems = document.querySelector(".detail-all-box");
            detailCheckItems.style.display = "block";
            clickArrorwsUp[i].style.display = "block";
            clickArrorwsDown[i].style.display = "none";
        }
    });
});

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// 파일 업로드

// 기본 썸네일 확인
const input = document.getElementById("input-thumnail");
const thumbnail = document.querySelector(".cover-image");
const cancel = document.querySelector(".cancel-image");
let fileBoxAdd = document.querySelector(".file-box-add");
let fileBoxAddSize = document.querySelector(".thumnail-size-text");

input.addEventListener("change", (e) => {
    let [file] = e.target.files;
    let reader = new FileReader();
    reader.readAsDataURL(file);
    reader.addEventListener("load", (e) => {
        const path = e.target.result;
        cancel.style.display = "block";
        if (path.includes("image")) {
            thumbnail.style.backgroundImage = `url(${path})`;
            fileBoxAdd.style.display = "none";
            fileBoxAddSize.style.display = "none";
        } else {
            thumbnail.style.backgroundImage = `url("https://eventusstorage.blob.core.windows.net/evsnull")`;
        }
    });
});
// x 버튼 클릭 시 삭제
cancel.addEventListener("click", (e) => {
    thumbnail.style.backgroundImage = `url("https://eventusstorage.blob.core.windows.net/evsnull")`;
    cancel.style.display = "none";
    fileBoxAdd.style.display = "block";
    fileBoxAddSize.style.display = "block";
    input.value = "";
});

// 추가 베너 확인
const inputExpand = document.getElementById("input-expand-thumnail");
const thumbnailExpand = document.querySelector(".expand-cover-image");
const cancelExpand = document.querySelector(".cancel-expand-image");

let fileBoxAddExpand = document.querySelector(".expand-file-box-add");
let fileBoxAddSizeExpand = document.querySelector(".thumnail-size-text-expand");

inputExpand.addEventListener("change", (e) => {
    let [file] = e.target.files;
    let readerExpand = new FileReader();
    console.log(file);
    readerExpand.readAsDataURL(file);
    readerExpand.addEventListener("load", (e) => {
        const pathExpand = e.target.result;
        cancelExpand.style.display = "block";
        if (pathExpand.includes("image")) {
            thumbnailExpand.style.backgroundImage = `url(${pathExpand})`;
            fileBoxAddExpand.style.display = "none";
            fileBoxAddSizeExpand.style.display = "none";
        } else {
            thumbnailExpand.style.backgroundImage = `url("https://eventusstorage.blob.core.windows.net/evsnull")`;
        }
    });
});

cancelExpand.addEventListener("click", (e) => {
    thumbnailExpand.style.backgroundImage = `url("https://eventusstorage.blob.core.windows.net/evsnull")`;
    cancelExpand.style.display = "none";
    fileBoxAddExpand.style.display = "block";
    fileBoxAddSizeExpand.style.display = "block";
    inputExpand.value = "";
});

if(activityimgge){
        cancel.style.display = "block";
        fileBoxAdd.style.display = "none";
        fileBoxAddSize.style.display = "none";
}


if(activitybaanerimg){
     cancelExpand.style.display = "block";
     fileBoxAddExpand.style.display = "none";
     fileBoxAddSizeExpand.style.display = "none";

}