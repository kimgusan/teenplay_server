// 헤더 사람 아이콘 클릭 시 마이페이지 간이 목록 표시 및 제거
const menuModalContainer = document.querySelector(".menu-modal-container");

document.addEventListener("click", (e) => {
    if (!e.target.closest(".mypage-menu-container")) {
        menuModalContainer.classList.remove("display:block");
    } else {
        if (e.target.closest(".menu-btn-box")) {
            menuModalContainer.classList.toggle("display:block");
        }
    }
    updateButtonState();
});

// 동의 체크박스 클릭 시 상태에 따라 경고 메시지 표시하는 이벤트
const inputCheckBox = document.querySelector("input[type=checkbox]");
const errorBoxs = document.querySelectorAll(".error-box");
const select = document.querySelector(".select");

inputCheckBox.addEventListener("change", (e) => {
    if (!e.target.checked) {
        errorBoxs[0].innerHTML = `<div id="agree-check-error" class="error">동의에 체크해주세요</div>`;
    } else {
        errorBoxs[0].innerHTML = "";
    }
    updateButtonState();
});

select.addEventListener("change", (e) => {
    errorBoxs[1].innerHTML = "";
});

//
const withdrawalBtn = document.querySelector(".withdrawal-btn");

withdrawalBtn.addEventListener("click", () => {
    // 체크박스가 체크되어 있는지 확인합니다.
    if (!inputCheckBox.checked) {
        errorBoxs[0].innerHTML = `<div id="agree-check-error" class="error">동의에 체크해주세요</div>`;
    } else {
        errorBoxs[0].innerHTML = ''; // 체크되어 있다면 이전에 있던 오류 메시지를 지웁니다.
    }

    // 선택된 값을 확인합니다.
    console.log(select.value);
    if (!select.value) {
        errorBoxs[1].innerHTML = `<div id="agree-check-error" class="error">값을 선택해주세요</div>`;
    } else {
        errorBoxs[1].innerHTML = ''; // 값을 선택했다면 이전에 있던 오류 메시지를 지웁니다.
    }

    // 모든 입력 필드가 채워져 있으면 버튼을 활성화합니다.
updateButtonState();
});

function updateButtonState() {
    if (inputCheckBox.checked && select.value) {
        // 모든 입력 필드가 채워져 있으면 버튼을 활성화하고 스타일을 변경합니다.
        withdrawalBtn.disabled = false;
        withdrawalBtn.style.backgroundColor = "#ce201b"; // 배경색 변경
        withdrawalBtn.style.color = "#ffffff"; // 텍스트 색상 변경
    } else {
        // 모든 입력 필드가 채워져 있지 않으면 버튼을 비활성화하고 스타일을 변경합니다.
        withdrawalBtn.disabled = true;
        withdrawalBtn.style.backgroundColor = "#F2F4F5"; // 배경색 변경
        withdrawalBtn.style.color = "#000000"; // 텍스트 색상 변경
    }
}

// 페이지 로드시 초기 버튼 상태를 업데이트합니다.
updateButtonState();