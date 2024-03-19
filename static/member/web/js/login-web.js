// // 로그인 버튼 쿼리
// const loginButton = document.querySelector(".login-button");
//
// // 아이디 미입력 이벤트
// // 아이디 입력 쿼리
// const idInputBox = document.querySelector(".id-input-container");
// // 비밀번호 입력 쿼리
// const passwordInputBox =document.querySelector(".password-input-container");
// // 아이디, 비밀번호 담기
// const boxCheck = (idInputBox && passwordInputBox)
//
// // 아이디 비밀번호 입력 이벤트
// // 아이디, 비밀번호 입력 시 이벤트 발생
// // 값이 있으면 로그인 버튼 활성화
// // 값이 없으면 비활성화
// boxCheck.addEventListener("keyup", () => {
//     if (boxCheck.value) {
//         loginButton.classList.remove("disabled");
//         return;
//     }
//     if (!boxCheck.value) {
//         loginButton.classList.add("disabled");
//     }
// })
//
// // 로그인 유지 클릭 시 메세지 출력 이벤트
// // 로그인 유지 버튼 쿼리
// const keepCheckBox = document.querySelector(".keep-checkbox");
// // 로그인 유지 관련 경고 메시지 쿼리
// const keepErrorMessage = document.querySelector(".warning-message-wrap");
//
// // 로그인 유지 버튼 클릭 시 경고 메시지 활성화 이벤트 발생
// keepCheckBox.addEventListener("click", () => {
//     keepErrorMessage.classList.toggle("hidden");
// });
