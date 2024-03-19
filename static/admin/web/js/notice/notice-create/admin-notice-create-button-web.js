// 버튼 업데이트 함수
function updateButtonStatus() {
    // 공지사항 제목 및 내용 입력창 가져오기
    let inputElements = document.querySelectorAll(".main-festival-info-detail-textarea, .main-festival-info-detail-input");
    // 공지 타입 라디오 버튼 가져오기
    let radioButtons = document.querySelectorAll("input[name='selection']");
    // 작성하기 버튼
    let createButton = document.querySelector(".main-festival-add-top-create-button");

    // 공지사항 제목 및 내용 입력창이 하나라도 비어있는지 확인
    let isAnyInputEmpty = Array.from(inputElements).some((input) => input.value.trim() === "");
    // 공지 타입 라디오 버튼이 하나라도 선택되었는지 확인
    let isAnyRadioButtonChecked = Array.from(radioButtons).some((radio) => radio.checked);

    //공지사항 입력창 및 버튼이 하나라도 없는 순간 작성하기 버튼 비활성화
    // createButton.disabled = isAnyInputEmpty || !isAnyRadioButtonChecked;
    // 비활성화 시 버튼 색상 변경
    createButton.style.backgroundColor = isAnyInputEmpty || !isAnyRadioButtonChecked ? "#f3f3f4" : "#CE201B";
    // 비활성화 시 마우스 스타일 변경
    createButton.style.cursor = isAnyInputEmpty || !isAnyRadioButtonChecked ? "default" : "pointer";
    // 비활성화 시 버튼 텍스트 색상 변경
    createButton.style.color = isAnyInputEmpty || !isAnyRadioButtonChecked ? "#b9b9bb" : "#fff";
}

// 페이지 로딩 시 라디오 버튼 과 라벨에 대한 이벤트 리스너 추가
document.addEventListener("DOMContentLoaded", () => {
    // 공지 타입 라디오 버튼 가져오기
    let radioButtons = document.querySelectorAll("input[name='selection']");
    // 공지 타입(공지사항, 자주묻는질문) 라벨 가져오기
    let labels = document.querySelectorAll("label[for='radio-notice'], label[for='radio-qr']");
    // 공지 타입 버튼 박스
    let boxes = document.querySelectorAll(".main-festival-box");

    // 공지 타입 라디오 반복하여 하나씩 가져오기
    radioButtons.forEach((radio) => {
        // 각 라디오 버튼에 대한 이벤트 리스너 추가
        // 값 변경 시 버튼 상태 변경(색상 등...)
        radio.addEventListener("change", updateButtonStatus);
    });

    // 라벨 반복하여 하나씩 가져오기
    labels.forEach((label) => {
        // 라벨 선택 시 이벤트 리스터 추가
        label.addEventListener("click", (e) => {
            // 라디오 아이디 가져오기
            let radioId = e.target.getAttribute("data-id");
            console.log(radioID)
            // 라디오 아이디 radio 변수에 담기
            let radio = document.getElementById(radioId);
            console.log(radio)
            // 만약, 라디오 값이 있으면
            if (radio) {
                // 라디오 체크 후 버튼 상태 업데이트
                radio.checked = true;
                updateButtonStatus();
            }
        });
    });

    boxes.forEach((box) => {
        box.addEventListener("click", function () {
            let radio = this.querySelector("input[name='selection']");
            if (radio) {
                radio.checked = true;
                updateButtonStatus();
            }
        });
    });

    // 입력 필드에 대한 이벤트 리스너를 추가합니다.
    let inputElements = document.querySelectorAll(".main-festival-info-detail-textarea, .main-festival-info-detail-input");
    // 입력 필드 입력 시 버튼 상태 변경
    inputElements.forEach((input) => {
        input.addEventListener("input", updateButtonStatus);
    });
});

// 라디오 버튼을 선택하고 선택된 박스의 스타일 변경
function checkRadioAndStyle(clickedBox, id) {
    // 라디오 버튼 박스 가져오기
    const boxes = document.querySelectorAll(".main-festival-box");

    // 반복문 사용
    // 박스 길이보다 i가 작으면 i증가
    // 박스 길이 미만으로 반복 (박스 2개) 0 < 2 (1번), 1 < 2 (2번), 2 < 2 (반복하지 않음)
    for (var i = 0; i < boxes.length; i++) {
        // 모든 박스의 selected-box클래스명 삭제
        boxes[i].classList.remove("selected-box");
    }

    // 클릭한 박스의 스타일 변경
    // 시각적으로 선택된 상태
    clickedBox.classList.add("selected-box");

    // 해당하는 라디오 버튼 선택
    // 클릭된 박스에 해당하는 라디오 버튼의 id
    // 선택된 라디오 버튼의 속성을 true로 설정하여 라디오 버튼 선택상태로 만들기
    document.getElementById(id).checked = true;
}
