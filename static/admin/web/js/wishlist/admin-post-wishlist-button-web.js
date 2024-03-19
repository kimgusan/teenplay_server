function updateButtonStatus() {
    let inputElements = document.querySelectorAll(
        ".admin-post-modal-title-input"
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
