// 결재 취소 클릭 시 작성 모달 여는 이벤트
const cancleModalWrap = document.querySelector(".cancle-modal-wrap");
const paymentCancle = document.querySelector(".payment-cancle");

paymentCancle.addEventListener("click", () => {
    cancleModalWrap.querySelector(".cancle-modal-container").style.animation = "popUp 0.5s";
    cancleModalWrap.style.display = "block";
});

// 결제 취소 모달 내 버튼 클릭 시 발생하는 이벤트
const cancleModalBtns = document.querySelectorAll(".cancle-modal-container button");
const textarea = cancleModalWrap.querySelector("textarea[name=cancle-content]");

cancleModalBtns.forEach((cancleModalBtn) => {
    cancleModalBtn.addEventListener("click", (e) => {
        if (e.target.className == "cancle-check-btn") {
            cancleModalWrap.querySelector(".cancle-modal-container").style.animation = "popDown 0.5s";
            setTimeout(() => {
                cancleModalWrap.querySelector(".cancle-modal-container").style.display = "none";
                cancleModalWrap.querySelector(".check-modal-container").style.animation = "popUp 0.5s";
                cancleModalWrap.querySelector(".check-modal-container").style.display = "flex";
                textarea.value = ``;
                checkValue();
            }, 450);
        } else {
            cancleModalWrap.querySelector(".cancle-modal-container").style.animation = "popDown 0.5s";
            setTimeout(() => {
                cancleModalWrap.style.display = "none";
                textarea.value = ``;
                checkValue();
            }, 450);
        }
    });
});

// 보내기 확인 모달 내 확인 클릭 시 모달 종료하는 이벤트
const checkModalCheckBtn = document.querySelector(".check-modal-container .check-btn");

checkModalCheckBtn.addEventListener("click", () => {
    cancleModalWrap.querySelector(".check-modal-container").style.animation = "popDown 0.5s";
    setTimeout(() => {
        cancleModalWrap.style.display = "none";
        cancleModalWrap.querySelector(".check-modal-container").style.display = "none";
        cancleModalWrap.querySelector(".cancle-modal-container").style.display = "flex";
    }, 450);
});

// textarea에 입력값이 있을 때 발생하는 이벤트
const cancleCheckBtn = document.querySelector(".cancle-check-btn");

textarea.addEventListener("input", () => {
    checkValue();
});

const checkValue = () => {
    if (textarea.value) {
        cancleCheckBtn.disabled = false;
        return;
    }
    cancleCheckBtn.disabled = true;
};
