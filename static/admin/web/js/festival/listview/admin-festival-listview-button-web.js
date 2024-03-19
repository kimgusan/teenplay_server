function updateButtonStatus() {
    let inputElements = document.querySelectorAll(
        ".admin-post-modal-title-input, .admin-post-modal-content-input, .admin-post-modal-pay-input, .admin-post-modal-adress-input, .admin-post-modal-adressadd-input, .admin-post-modal-place-input, .admin-post-modal-sttime-input, .admin-post-modal-endtime-input, .admin-post-modal-time-input, .admin-post-modal-opne-input, .admin-post-modal-phone-input, .admin-post-modal-give-input, .admin-post-modal-url-input"
    );
    let buttonElement = document.querySelector(
        ".admin-user-modal-right-detail-button"
    );

    let isAnyInputEmpty = Array.from(inputElements).some(
        (input) => input.value.trim() === ""
    );
    buttonElement.disabled = isAnyInputEmpty;
    buttonElement.style.backgroundColor = isAnyInputEmpty
        ? "#f3f3f4"
        : "#CE201B";
    buttonElement.style.cursor = isAnyInputEmpty ? "default" : "pointer";
    buttonElement.style.color = isAnyInputEmpty ? "#b9b9bb" : "#fff";
}
