const button = document.querySelector(
    ".main-festival-content-mid-container-button"
);

const content = document.querySelector(
    ".main-festival-content-mid-container-in"
);

button.addEventListener("click", () => {
    content.classList.toggle("hidden");
});
