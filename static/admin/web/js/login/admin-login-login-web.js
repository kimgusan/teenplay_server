const button = document.querySelector(".main-login-button");
const id = document.querySelector(".main-login-id");
const pw = document.querySelector(".main-login-pw");
const modal = document.querySelector("#admin-message-modal");
const backmodal = document.querySelector("#admin-message-modal-backdrop");

const canclebutton = document.querySelector(".admin-message-modal-left-button");

// // 로그인 정보 검사 후 페이지 이동
// button.addEventListener("click", () => {
//     // 임시? 아이디비번
//     if ((id.value === "teenplay" && pw.value === "1234")) {
//         // 페이지 나오면 페이지이동 url
//         window.location.href = "";
//         return;
//     }
//     modal.classList.remove("hidden");
//     backmodal.classList.remove("hidden");
// });


// 로그인 버튼 클릭 시 success() 함수 실행
button.addEventListener("click", () => {
    success();
})

// enter 클릭 시 success() 함수 실행
document.addEventListener("keyup", (evnet) => {
    if (evnet.keyCode === 13) {
        success();
    }
})

// 아이디, 비밀번호 검사 후 관리자 정보와 일치하면
// /admin/user/ 페이지로 이동
function success() {
    if ((id.value === "teenplay" && pw.value === "1234")) {

        window.location.href = "/admin/user/";
        return;
    }
    modal.classList.remove("hidden");
    backmodal.classList.remove("hidden");
}

// 회원정보 오입력 모달 닫기 버튼 클릭 시 모달 종료
canclebutton.addEventListener("click", () => {
    modal.classList.add("hidden");
    backmodal.classList.add("hidden");
});

//회원정보 오입력 모달 화면 클릭 시 모달 종료
document.addEventListener("click", (e) => {
    if (!button.contains(e.target) && !modal.contains(e.target)) {
        modal.classList.add("hidden");
        backmodal.classList.add("hidden");
    }
})

