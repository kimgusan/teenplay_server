// 1단계 - 모임 이름 입력 검사 (버튼 활성화)
const nameInput = document.querySelector("input.name-form-input");
const nameButton = document.querySelector("button.name-form-button");
const categorySelect = document.querySelector(".pr-write-fields");
const interestSelectBox = document.querySelector(".interest-select-box");
const categoryCheckboxes = document.querySelectorAll(".hidden-checkbox");
const selectError = document.querySelector(".select-error")
const categoryLabels = document.querySelectorAll(".interest-category")

// 버튼 클릭 시 한번 더 검사하기 위해
let flag = false;

let categoryValue = false;
let nameValue = false;

// 대표 카테고리 입력 검사
categorySelect.addEventListener("change", (e) => {
    if (e.target.value) {
        categoryValue = true;
        const categorySelectValue = categorySelect.value

        // 선택된 대표 카테고리를 추가 카테고리 항목에서 없애기
        categoryLabels.forEach((label) => {
            if (label.getAttribute('for') === categorySelectValue) {
                label.style.display = "none";
            } else {
                label.style.display = 'block';
            }
        })
    } else {
        categoryValue = false;
    }
    allCheck();
})

// 모임명 입력 검사
nameInput.addEventListener("keyup", (e) => {
    if (e.target.value) {
        nameValue = true;
    } else {
        nameValue = false;
    }
    allCheck();
})

// 카테고리 체크 검사
let categoryCheckValue = true;
let checkedCount = 0;

categoryCheckboxes.forEach((checkbox) => {
    checkbox.addEventListener("click", () => {
        checkedCount = Array.from(categoryCheckboxes).filter((checkbox) => checkbox.checked).length
        if (checkedCount < 4) {
            selectError.classList.add("hidden")
            categoryCheckValue = true;
        } else {
            selectError.classList.remove("hidden")
            categoryCheckValue = false;
        }
        allCheck();
    })
})

// 전체 입력 시 활성화
function allCheck() {
    if (categoryValue && nameValue) {
        interestSelectBox.classList.remove("hidden");
        nameButton.classList.remove("disabled");
        flag = true;

        if (categoryCheckValue === false) {
            nameButton.classList.add("disabled");
            flag = false;
        }
        return;
    }
    nameButton.classList.add("disabled");
    flag = false;
}

// 작성한 모임명을 각 태그 안에 넣어 주는 함수
const writeClubName = () => {
    const nameInput = document.querySelector("input.name-form-input");
    document.querySelector(".desc-form-text-title").innerText = `[${nameInput.value}]에 대한 소개글과\n활동 지역을 선택해주세요.`;
    document.querySelector(".img-form-text-title").innerText = `[${nameInput.value}] 모임을 고유한 이미지로 꾸며주세요`;
    document.querySelector(".club-detail-name").innerText = `${nameInput.value}`

}

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

// 입력창 가져오기
const nameInputBox = document.querySelector(".name-form-input-box");

// 입력창 클릭 시 입력박스 색상 변경
nameInputBox.addEventListener("click", (event) => {
    nameInputBox.style.outline = "#2563EB solid 2px";
    // 이벤트 전파를 막아 문서 전체의 클릭 이벤트가 트리거되지 않도록 합니다.
    event.stopPropagation();
})

// 입력창 외 화면 클릭 시 색상 변경 제거
document.addEventListener("click", () => {
    nameInputBox.style.outline = "";
});


// 다음 버튼 클릭 시 내용 변경(2단계로 이동)

const firstContent = document.querySelector(".name-form-wrap");
const secondContent = document.querySelector(".desc-form-wrap");
nameButton.addEventListener("click", () => {
    firstContent.style.display = "none";
    secondContent.style.display = "block";
    writeClubName()
});

// 2단계
// 이전 단계 버튼 클릭 시 다시 1단계로

const prevButton = document.querySelector(".desc-form-prev-button");
prevButton.addEventListener("click", () => {
    firstContent.style.display = "block";
    secondContent.style.display = "none";
});

// 소개글 입력 박스 가져오기
const descInputBox = document.querySelector(".desc-form-input-box");

// 소개글 입력창 클릭 시 입력박스 테두리 색상 변경
descInputBox.addEventListener("click", (event) => {
    descInputBox.style.outline = "#2563EB solid 2px";
    // 이벤트 전파를 막아 문서 전체의 클릭 이벤트가 트리거되지 않도록 합니다.
    event.stopPropagation();
})

// 소개글 입력창 외 화면 클릭 시 테두리 색상 변경 제거
document.addEventListener("click", () => {
    descInputBox.style.outline = "";
});

// 소개글 입력 검사
const descInput = document.querySelector("input.desc-form-input");
const descButton = document.querySelector(".desc-form-button");

// descInput.addEventListener("keyup", () => {
//     if (descInput.value) {
//         descButton.classList.remove("disabled");
//         return;
//     }
//     if (!descButton.classList.contains("disabled")) {
//         descButton.classList.add("disabled");
//     }
// });

// 지역 선택 검사
let regionValue = false;
const regionSelect = document.querySelector(".pr-write-desc-fields")

// 대표 카테고리 입력 검사
regionSelect.addEventListener("change", (e) => {
    if (e.target.value) {
        regionValue = true;
    } else {
        regionValue = false;
    }
    descCheck();
})

let descValue = false;

// 소개글 입력 검사
descInput.addEventListener("keyup", (e) => {
    if (e.target.value) {
        descValue = true;
    } else {
        descValue = false;
    }
    descCheck();
})

// 전체 입력 시 활성화
let flagDE = false;

function descCheck() {
    if (regionValue && descValue) {
        descButton.classList.remove("disabled");
        flagDE = true;
        return;
    }
    descButton.classList.add("disabled");
    flagDE = false;
}

// 작성한 모임 한 줄 소개를 태그 안에 넣어 주는 함수
const writeClubIntro = () => {
    const descInput = document.querySelector("input.desc-form-input");
    document.querySelector(".club-detail-introduction").innerText = `${descInput.value}`
}

// 다음 버튼 클릭 시 내용 변경(3단계로 이동)

const thirdContent = document.querySelector(".img-form-wrap");
descButton.addEventListener("click", () => {
    secondContent.style.display = "none";
    thirdContent.style.display = "block";
    writeClubIntro()
});

// 3단계
// 이전 단계 버튼 클릭 시 다시 2단계로

const lastPrevButton = document.querySelector(".img-form-prev-button");
lastPrevButton.addEventListener("click", () => {
    secondContent.style.display = "block";
    thirdContent.style.display = "none";
});

// 개설 완료 및 건너뛰기 버튼은 서버 작업 시 연결.

// 아래는 이미지 첨부 부분입니다.
// 이미지 썸네일을 화면에 표시하는 부분은 서버 담당 시 구현합니다.
// 우선은 그대로 표시하겠습니다.

// "모임 프로필 업로드" 버튼 클릭 시 프로필 사진 input 활성화(파일 업로드)
// 및 파일 용량 체크 (프사, 커버 10MB 제한)
function checkFileSize(obj, size) {
    let check = false;
    let sizeInBytes = obj.files[0].size;
    if (sizeInBytes > size) {
        check = false;
    } else {
        check = true;
    }
    return check;
}

function getFileSizeWithExtension(sizeInBytes) {
    let fileSizeExt = new Array("bytes", "kb", "mb", "gb");
    let i = 0;
    let checkSize = sizeInBytes;
    while (checkSize > 900) {
        checkSize /= 1024;
        i++;
    }
    checkSize = Math.round(checkSize * 100) / 100 + "" + fileSizeExt[i];
    return checkSize;
}

const MAX_SIZE = 10; // 10MB

const openProfile = () => {
    document.getElementById("profile-image").click();
};

const profileInput = document.getElementById("profile-image");
const sizeErrorMsg = document.querySelector(".img-form-profile-size-error");
const profileImage = document.querySelector(".img-form-thumbnail");
const profileBackground = document.querySelector(".img-form-thumbnail-bg");
const profileDeleteButton = document.querySelector(".profile-delete-wrap");
const profileDeleteIconDefault = document.querySelector(".profile-delete");
const profileDeleteIconHover = document.querySelector(".profile-delete-hover");

// 삭제 버튼 hover 이벤트마다 색상 변경부터
profileDeleteButton.addEventListener("mouseover", () => {
    profileDeleteIconDefault.style.display = "none";
    profileDeleteIconHover.style.display = "block";
});

profileDeleteButton.addEventListener("mouseout", () => {
    profileDeleteIconDefault.style.display = "block";
    profileDeleteIconHover.style.display = "none";
});

profileDeleteButton.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    sizeErrorMsg.style.display = "none";
    profileInput.value = "";
    profileImage.setAttribute("src", "https://event-us.kr/Content/neweventus/image/hostcenter/hostcenter_create_upload_01.png");
    profileBackground.style.display = "none";
    // 삭제 버튼 안 보이게 하기
    profileDeleteButton.style.display = "none";
});

profileInput.addEventListener("change", (e) => {
    if (e.target.value) {
        let checkSize = 1024 * 1024 * MAX_SIZE;
        if (!checkFileSize(profileInput, checkSize)) {
            sizeErrorMsg.style.display = "block";
            e.preventDefault();
            return;
        }
        sizeErrorMsg.style.display = "none";
        // 서버 작업은 여기에 fetch로 작성한 후 썸네일을 받아와 화면에 표시합니다.
        // 우선은 올린 이미지를 그대로 표시하겠습니다.
        let reader = new FileReader();
        reader.onload = (e) => {
            profileImage.setAttribute("src", e.target.result);
            profileBackground.style.display = "block";
            // 삭제 버튼 보이게 하기
            profileDeleteButton.style.display = "inline-block";
        };
        reader.readAsDataURL(e.target.files[0]);
    }
});

// "모임 커버 업로드" 버튼 클릭 시 커버 사진 input 활성화(파일 업로드)
const coverUploadInput = document.getElementById("background-image");
const openCover = () => {
    coverUploadInput.click();
};

// 업로드 시 사이즈 체크 및 썸네일 표시
const coverImage = document.querySelector(".cover-thumbnail");
const coverUploadWrap = document.querySelector(".cover-upload-wrap");
const uploadSpanGray = document.querySelector(".upload-span-gray");
const coverBackground = document.querySelector(".cover-thumbnail-container");
const coverDeleteButton = document.querySelector(".cover-delete-wrap");
const coverDeleteIconDefault = document.querySelector(".cover-delete");
const coverDeleteIconHover = document.querySelector(".cover-delete-hover");

// 삭제 버튼 hover 이벤트마다 색상 변경부터
coverDeleteButton.addEventListener("mouseover", () => {
    coverDeleteIconDefault.style.display = "none";
    coverDeleteIconHover.style.display = "block";
});

coverDeleteButton.addEventListener("mouseout", () => {
    coverDeleteIconDefault.style.display = "block";
    coverDeleteIconHover.style.display = "none";
});

coverDeleteButton.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    sizeErrorMsg.style.display = "none";
    coverImage.setAttribute("src", "https://event-us.kr/Content/neweventus/image/hostcenter/hostcenter_create_upload_01.png");
    coverUploadInput.value = "";
    // 글들 다 보이게
    coverUploadWrap.style.display = "block";
    uploadSpanGray.style.display = "block";
    // 배경 지우기
    coverBackground.style.display = "none";
    // 삭제 버튼 안 보이게 하기
    coverDeleteButton.style.display = "none";
});

coverUploadInput.addEventListener("change", (e) => {
    if (e.target.value) {
        let checkSize = 1024 * 1024 * MAX_SIZE;
        if (!checkFileSize(coverUploadInput, checkSize)) {
            sizeErrorMsg.style.display = "block";
            e.preventDefault();
            return;
        }
        sizeErrorMsg.style.display = "none";
        // 서버 작업은 여기에 fetch로 작성한 후 썸네일을 받아와 화면에 표시합니다.
        // 우선은 올린 이미지를 그대로 표시하겠습니다.
        let reader = new FileReader();
        reader.onload = (e) => {
            coverImage.setAttribute("src", e.target.result);
            // 글들 다 지우기
            coverUploadWrap.style.display = "none";
            uploadSpanGray.style.display = "none";
            // 배경 투명한 어두운 색으로 삭제 버튼 잘 보이도록 하기
            coverBackground.style.display = "block";
            // 삭제 버튼 보이게 하기
            coverDeleteButton.style.display = "inline-block";
        };
        reader.readAsDataURL(e.target.files[0]);
    }
});

// 드래그 앤 드롭으로 커버 업로드하기

const dragDropBox = document.querySelector(".cover-thumbnail-wrap");
dragDropBox.addEventListener("dragenter", (e) => {
    e.preventDefault();
});
dragDropBox.addEventListener("dragover", (e) => {
    e.preventDefault();
});
dragDropBox.addEventListener("dragleave", (e) => {
    e.preventDefault();
});
dragDropBox.addEventListener("drop", (e) => {
    e.preventDefault();
    let file = e.dataTransfer;
    if (!checkFileSize(file, 1024 * 1024 * MAX_SIZE)) {
        sizeErrorMsg.style.display = "block";
        return;
    }
    sizeErrorMsg.style.display = "none";
    // 서버 작업은 여기에 fetch로 작성한 후 썸네일을 받아와 화면에 표시합니다.
});

const imgFormBtn = document.querySelector('.img-form-button');
const clubCreateForm = document.querySelector('form[name=club-create-form]')
imgFormBtn.addEventListener("click", (e) => {
    clubCreateForm.submit()
})