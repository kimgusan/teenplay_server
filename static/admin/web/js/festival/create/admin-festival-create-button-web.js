function updateButtonStatus() {
    let inputElements = document.querySelectorAll(
        ".main-festival-info-detail-input, .main-festival-plan-detail-pay-input, .main-festival-plan-detail-adress-input, .main-festival-plan-detail-adressadd-input, .main-festival-plan-detail-place-input, .main-festival-plan-detail-startdate-input, .main-festival-plan-detail-lastdate-input, .main-festival-plan-detail-time-input, .main-festival-admin-detail-info-input, .main-festival-admin-detail-phone-input, .main-festival-give-detail-info-input, .main-festival-give-detail-phone-input, .main-festival-info-detail-textarea, .main-festival-photo-input"
    );
    let buttonElement = document.querySelector(
        ".main-festival-add-top-create-button"
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
