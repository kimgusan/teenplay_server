const writeBtn = document.querySelector(".notice-write-button");
const titleInput = document.querySelector(".notice-write-title");
const contentInput = document.querySelector(".notice-write-content");

// 카테고리, 제목, 내용, 이미지 다 입력했을 때에만 작성완료 버튼 활성화
// 이미지 부분은 아래에 있습니다.
const checkValuesAndValidateButton = () => {
    if (!titleInput.value || !contentInput.value) {
        writeBtn.disabled = true;
        return;
    }
    writeBtn.disabled = false;
};

titleInput.addEventListener("input", () => {
    checkValuesAndValidateButton();
});

contentInput.addEventListener("input", () => {
    checkValuesAndValidateButton();
});

checkValuesAndValidateButton();

// 전부 다 입력 후 서버쪽 작성완료 구현은 아래 주석한 것처럼 하면 됩니다.

// writeBtn.addEventListener("click", () => {
//     if (조건이 다 맞는다면) {
//         document.write.submit(); // form 태그를 submit()시킵니다.
//     }
// })
