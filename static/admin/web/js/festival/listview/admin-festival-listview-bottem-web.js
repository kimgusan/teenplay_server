// HTML 요소들을 가져옵니다.
const buttons = document.querySelectorAll(".main-user-bottom");

// 각 버튼을 클릭했을 때 실행될 함수를 정의합니다.
buttons.forEach((button, index) => {
    button.addEventListener("click", function () {
        // 모든 버튼에서 add-color 클래스를 제거합니다.
        buttons.forEach((btn) => btn.classList.remove("add-color"));
        // 현재 클릭한 버튼에 add-color 클래스를 추가합니다.
        button.classList.add("add-color");
        // 모든 버튼의 span 요소의 글자색을 초기화합니다.
        document
            .querySelectorAll(".main-user-number")
            .forEach((span) => (span.style.color = ""));
        // 현재 클릭한 버튼의 span 요소의 글자색을 변경합니다.
        document.querySelector(
            `#button${index + 1} .main-user-number`
        ).style.color = "#fff";
    });
    // 페이지 로딩 시 초기화할 때 1번 버튼의 글자색도 #fff로 변경합니다.
    if (index === 0) {
        document.querySelector(
            `#button${index + 1} .main-user-number`
        ).style.color = "#fff";
    }
});
