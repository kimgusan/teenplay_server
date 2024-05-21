const categorySelect = document.querySelector(".pr-write-fields");
const categoryLabels = document.querySelectorAll(".interest-category")
const categoryCheckboxes = document.querySelectorAll(".hidden-checkbox");
const selectError = document.querySelector(".club-sub-category-warning")

// 버튼 클릭 시 한번 더 검사하기 위해
let flag = false;
let categoryValue = false;

// 지역 선택 검사
let regionValue = false;
const regionSelect = document.querySelector(".pr-write-desc-fields")

// 카테고리 체크 검사
let categoryCheckValue = true;
let checkedCount = 0;

// club-name-input의 글자수 표시하는 이벤트
const clubNameInput = document.querySelector(".club-name-input");
const inputValueCount = document.querySelector(".input-value-count");

inputValueCount.innerText = `${clubNameInput.value.length} / 20`;

clubNameInput.addEventListener("input", () => {
    inputValueCount.innerText = `${clubNameInput.value.length} / 20`;
});

// 모임 프로필 업로드 클릭 시 label이 클릭 되는 이벤트
const clubProfileUploadBtn = document.querySelector(".club-profile-upload-btn");
const clubProfileUploadLabel = document.querySelector(".club-profile-input-label");
const clubProfileInput = document.querySelector("#club-profile-input");
const clubProfileThumbnail = document.querySelector('.club-profile-thumbnail')
clubProfileUploadBtn.addEventListener("click", () => {
    clubProfileUploadLabel.click();
});

clubProfileInput.addEventListener('change', (e) => {
    const targetInput = e.target;
    const file = targetInput.files[0];
    const reader = new FileReader();

    reader.onload = (event) => {
        const path = event.target.result;
        clubProfileThumbnail.setAttribute("src", path);
    };
    if (file) {
        reader.readAsDataURL(file);
    }
})

// 모임 프로필 업로드 시 이미지 확장자가 아니라면 오류 모달 표시

clubProfileInput.addEventListener("change", (e) => {
    if (e.target.value) {
        let fileName = e.target.files[0].name;

        // 이미지 관련 확장자가 아니라면 확장자 오류 모달 표시
        if (!checkExtension(fileName, exampeExtensions)) {
            e.target.value = "";
            extensionErrorModalContainer.style.animation = "popUp 0.5s";
            extensionErrorModalWrap.style.display = "block";
            return;
        }
    }
});

// 이미지 업로드 클릭 시 배경이미지 input이 클릭 되는 이벤트
const backgroundUploadBtn = document.querySelector(".background-upload-btn");
const backgroundImgInput = document.querySelector(".background-img-input");

backgroundUploadBtn.addEventListener("click", () => {
    backgroundImgInput.click();
});

const noneBackgroundWrap = document.querySelector(".none-background-wrap");
const prevImgBox = document.querySelector('.prev-img-box')
const prevCoverImg = () => {
    if (clubBannerPath) {
        noneBackgroundWrap.style.display = 'none'
        prevImgBox.style.display = 'block'
        const prevImgCancel = document.querySelector('.prev-img-cancel')
        prevImgCancel.addEventListener('click', () => {
            noneBackgroundWrap.style.display = 'block'
            prevImgBox.style.display = 'none'
            backgroundFileWrap.style.display = "none";
            backgroundImgInput.value = ''
        })
    } else {
        noneBackgroundWrap.style.display = 'block'
        prevImgBox.style.display = 'none'
        const backgroundUploadBtn = document.querySelector(".background-upload-btn");
        const backgroundImgInput = document.querySelector(".background-img-input");
        backgroundUploadBtn.addEventListener("click", () => {
            backgroundImgInput.click();
        });
    }
}
prevCoverImg()

const backgroundThumbnail = document.querySelector('.background-thumbnail')
const prevImgBoxFn = () => {
    backgroundImgInput.addEventListener('change', (e) => {
        const targetInput = e.target;
        const file = targetInput.files[0];
        const reader = new FileReader();

        reader.onload = (event) => {
            const path = event.target.result;
            backgroundThumbnail.setAttribute("src", path);
            noneBackgroundWrap.style.display = 'none'
            prevImgBox.style.display = 'block'

        };
        if (file) {
            reader.readAsDataURL(file);
        }
    })
}
prevImgBoxFn()
// 파일의 용량으로 단위를 계산하고 수정해주는 함수
const getFileSizeWithExtension = (sizeInBytes) => {
    let fileSizeExt = new Array("bytes", "kb", "mb", "gb");
    let i = 0;
    let checkSize = sizeInBytes;
    while (checkSize > 900) {
        checkSize /= 1024;
        i++;
    }
    checkSize = Math.round(checkSize * 100) / 100 + "" + fileSizeExt[i];
    return checkSize;
};

// 파일의 확장자를 확인해주는 함수
const checkExtension = (fileName, extensions) => {
    // 파일 이름에서 확장자를 추출
    const fileExtension = fileName.split(".").pop().toLowerCase();
    // 허용할 목록들과 비교
    return extensions.includes(fileExtension);
};

// 배경화면 input이 바뀔때 확장자를 검사하고 용량을 변환하여 뿌리는 이벤트
const exampeExtensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp"];
const extensionErrorModalWrap = document.querySelector(".extension-error-modal-wrap");
const extensionErrorModalContainer = extensionErrorModalWrap.querySelector(".extension-error-modal-container");
const backgroundFileCapacity = document.querySelector(".background-file-capacity");
const backgroundFileName = document.querySelector(".background-file-name");
const backgroundFileWrap = document.querySelector(".background-file-wrap");

const createBackgroundInfo = () => {
    if (backgroundImgInput.value) {
        let fileName = backgroundImgInput.files[0].name;

        // 이미지 관련 확장자가 아니라면 확장자 오류 모달 표시
        if (!checkExtension(fileName, exampeExtensions)) {
            backgroundImgInput.value = "";
            extensionErrorModalContainer.style.animation = "popUp 0.5s";
            extensionErrorModalWrap.style.display = "block";
            return;
        }

        fileSize = backgroundImgInput.files[0].size;
        backgroundFileCapacity.innerText = getFileSizeWithExtension(fileSize);
        backgroundFileName.innerText = fileName;
        backgroundFileWrap.style.display = "block";
    }
};

backgroundImgInput.addEventListener("change", () => {
    createBackgroundInfo();

});

// 확장자 오류 모달 닫는 이벤트
const extensionErrorModalCloseBtn = document.querySelector(".extension-error-modal-close-btn");

extensionErrorModalCloseBtn.addEventListener("click", () => {
    extensionErrorModalContainer.style.animation = "popDown 0.5s";
    setTimeout(() => {
        extensionErrorModalWrap.style.display = "none";
    }, 450);
});

// 배경화면 정보 삭제 클릭 시 발생하는 이벤트
const backgroundCancleBox = document.querySelector(".background-cancle-box");

backgroundCancleBox.addEventListener("click", () => {
    prevImgBox.style.display = 'none'
    backgroundFileWrap.style.display = "none";
    noneBackgroundWrap.style.display = 'block'
    backgroundImgInput.value = ''
});

// 모임명, 담당자 이름, 이메일, 전화번호의 상태에 따라 발생하는 이벤트
const clubManagerNameInput = document.querySelector("#club-manager-name-input");
const clubManagerEmailInput = document.querySelector("#club-manager-email-input");
const clubManagerPhoneInput = document.querySelector("#club-manager-phone-input");
const clubNameInputError = document.querySelector(".club-name-input-error");
const noneNameWarning = document.querySelector(".none-name-warning");
const noneEmailWarning = document.querySelector(".none-email-warning");
const nonePhoneWarning = document.querySelector(".none-phone-warning");
const clubSaveBtn = document.querySelector(".club-save-btn");

const checkInputValue = (checkedCount) => {
    console.log(checkedCount)
    if (!clubNameInput.value || !clubManagerNameInput.value || !clubManagerEmailInput.value || !clubManagerPhoneInput.value || !categorySelect.value || !regionSelect.value || (checkedCount > 3)) {
        clubSaveBtn.disabled = true;
        selectError.classList.remove("hidden")
        return;
    }
    if (!(checkedCount > 3)) {
        selectError.classList.add("hidden")
    }
    clubSaveBtn.disabled = false;
};
clubNameInput.addEventListener("input", () => {
    checkInputValue();
});
clubManagerNameInput.addEventListener("input", () => {
    checkInputValue();
});
clubManagerEmailInput.addEventListener("input", () => {
    checkInputValue();
});
clubManagerPhoneInput.addEventListener("input", () => {
    checkInputValue();
});
checkInputValue();


noneBackgroundWrap.addEventListener("dragenter", (e) => {
    e.preventDefault();
});
noneBackgroundWrap.addEventListener("dragover", (e) => {
    e.preventDefault();
});
noneBackgroundWrap.addEventListener("dragleave", (e) => {
    e.preventDefault();
});
noneBackgroundWrap.addEventListener("drop", (e) => {
    e.preventDefault();
    let file = e.dataTransfer;
    backgroundImgInput.files = file.files;
    createBackgroundInfo();
});

// 관심 분야 클릭 시 클래스 추가로 속성 변화
const interestCaregories = document.querySelectorAll(".interest-category");

interestCaregories.forEach((interestCaregory) => {
    interestCaregory.addEventListener("click", () => {
        if (interestCaregory.classList.contains("selected-category")) {
            interestCaregory.classList.remove("selected-category");
            return;
        }
        interestCaregory.classList.add("selected-category");
    });
});

// 대표 카테고리 입력 검사
categorySelect.addEventListener("change", (e) => {
    if (e.target.value) {
        categoryValue = true;
        const categorySelectValue = categorySelect.value

        // 선택된 대표 카테고리를 추가 카테고리 항목에서 없애기
        categoryLabels.forEach((label) => {
            if (label.getAttribute('for') === categorySelectValue) {
                label.style.display = "none";
                // 선택된 대표 카테고리와 동일한 추가 카테고리의 체크박스 선택 해제
                const checkbox = document.getElementById(categorySelectValue);
                if (checkbox) {
                    checkbox.checked = false;
                }
            } else {
                label.style.display = 'block';
            }
        })
    } else {
        categoryValue = false;
    }
    checkInputValue();
})

// 대표 카테고리 입력 검사
regionSelect.addEventListener("change", (e) => {
    if (e.target.value) {
        regionValue = true;
    } else {
        regionValue = false;
    }
    checkInputValue();
})

categoryCheckboxes.forEach((checkbox) => {
    checkbox.addEventListener("click", () => {
        checkedCount = Array.from(categoryCheckboxes).filter((checkbox) => checkbox.checked).length
        checkInputValue(checkedCount);
    })
})

document.addEventListener("DOMContentLoaded", () => {
    // 선택된 대표 카테고리를 추가 카테고리 항목에서 없애기
    const hideSelectedCategory = () => {
        const categorySelectValue = categorySelect.value;
        categoryLabels.forEach((label) => {
            if (label.getAttribute('for') === categorySelectValue) {
                label.style.display = "none";
            } else {
                label.style.display = 'block';
            }
        });
    };

    // 페이지 로드 시 숨기기 로직 실행
    hideSelectedCategory();
});