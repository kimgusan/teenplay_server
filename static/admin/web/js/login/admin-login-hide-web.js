function togglePasswordVisibility() {
    var passwordInput = document.querySelector(".main-login-pw");

    if (passwordInput.type === "password") {
        passwordInput.type = "text";
    } else {
        passwordInput.type = "password";
    }
}
