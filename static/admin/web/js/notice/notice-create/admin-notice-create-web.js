const button = document.querySelector(".main-festival-add-top-create-button");
const typeButtons = document.querySelectorAll("input[name=notice_type]");

const titleInput = document.querySelector("input[name=notice_title]");
const contentInput = document.querySelector("textarea[name=notice_content]");
function inputValue() {
    const typeButtons = document.querySelectorAll("input[name=notice_type]");

    let inputElements = document.querySelectorAll("input[name=notice_title], input[name=notice_content]");
    // let isAnyInputEmpty = Array.from(inputElements).some((input) => input.value.trim() === "");
    let isAnyRadioButtonChecked = Array.from(typeButtons).some((radio) => radio.checked);

    // 모든 입력 필드가 채워져 있고 라디오 버튼이 선택되었는지 확인합니다.
    // if (!isAnyInputEmpty && isAnyRadioButtonChecked) {
    if (titleInput.value && contentInput.value){
        // 모든 입력이 유효한 경우 버튼을 활성화합니다.
        button.classList.remove("disabled");
    } else {
        // 입력이 유효하지 않은 경우 버튼을 비활성화합니다.
        button.classList.add("disabled");
    }
}

titleInput.addEventListener("keyup", () => {
    if (titleInput.value && contentInput.value){
        // 모든 입력이 유효한 경우 버튼을 활성화합니다.
        button.classList.remove("disabled");
    } else {
        // 입력이 유효하지 않은 경우 버튼을 비활성화합니다.
        button.classList.add("disabled");
    }
})

contentInput.addEventListener("keyup", () => {
    if (titleInput.value && contentInput.value){
        // 모든 입력이 유효한 경우 버튼을 활성화합니다.
        button.classList.remove("disabled");
    } else {
        // 입력이 유효하지 않은 경우 버튼을 비활성화합니다.
        button.classList.add("disabled");
    }
})

typeButtons.forEach((e) => {
    e.addEventListener("click", () => {
        inputValue();
    });
})

button.addEventListener("click", (e) => {
    if (button.classList.contains("disabled")) {
        // 버튼이 비활성화된 경우 폼 제출을 중단합니다.
        // e.preventDefault();
    } else {
        // 버튼이 활성화된 경우 폼을 제출합니다.
        document['notice-create-form'].submit();
    }
});


const boxes = document.querySelectorAll(".main-festival-box");

boxes.forEach((box) => {
    const typeButton = box.querySelector("input[type=radio]");

    box.addEventListener('click', () => {
        // 모든 박스의 selected-box 클래스명 삭제
        boxes.forEach((b) => {
            b.classList.remove("selected-box");
        });

        box.classList.add("selected-box");

        // 박스 내의 라디오 버튼을 선택합니다.
        typeButton.checked = true;
    });
});
