// 변경 버튼
const changeBtn = document.querySelector(".change-btn");
// 회원 이름 보여주는 태그
const nicknameView = document.querySelector(".nickname-view");
// 이름 수정하는 태그
const nicknameModify = document.querySelector(".nickname-modify");
// 변경할 이름을 입력하는 input
const nicknameInput = document.querySelector(".nickname-input");
// nicknameView안의 span태그 안의 text를 저장
let nickname = nicknameView.firstElementChild;

// 변경 클릭 시 nicknameView을 감추고 nicknameModify을 보여주고 span의 text를 input의 value에 넣어준다.
changeBtn.addEventListener("click", () => {
    nicknameView.style.display = "none";
    nicknameModify.style.display = "";

    nicknameInput.value = nickname.innerText;
});

// 확인 버튼
const nicknameFix = document.querySelector(".nickname-fix");

// 확인 클릭 시 nicknameModify을 감추고 nicknameView을 보여주고 input의 value를 span에 넣어준다.
nicknameFix.addEventListener("click", () => {
    nicknameView.style.display = "flex";
    nicknameModify.style.display = "none";

    nickname.innerText = nicknameInput.value;
});

// 모임 프로필 업로드 클릭 시 label이 클릭 되는 이벤트
const profileUploadBtn = document.querySelector(".profile-upload-btn");
const profileUploadLabel = document.querySelector(".profile-input-label");
const profileInput = document.querySelector("#profile-input");

profileUploadBtn.addEventListener("click", () => {
    profileUploadLabel.click();
});

// 모임 프로필 업로드 시 이미지 확장자가 아니라면 오류 모달 표시
const exampeExtensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp"];
const extensionErrorModalWrap = document.querySelector(".extension-error-modal-wrap");
const extensionErrorModalContainer = extensionErrorModalWrap.querySelector(".extension-error-modal-container");

profileInput.addEventListener("change", (e) => {
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

function previewImage(input) {
    var previewThumbnail = document.getElementById('preview-thumbnail');
    var profileInput = document.getElementById('profile-input');

    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            previewThumbnail.src = e.target.result;
            profileInput.style.backgroundImage = "url('" + e.target.result + "')";
        };

        reader.readAsDataURL(input.files[0]);
    }
}


// 확장자 오류 모달 닫는 이벤트
const extensionErrorModalCloseBtn = document.querySelector(".extension-error-modal-close-btn");

extensionErrorModalCloseBtn.addEventListener("click", () => {
    extensionErrorModalContainer.style.animation = "popDown 0.5s";
    setTimeout(() => {
        extensionErrorModalWrap.style.display = "none";
    }, 450);
});

// 파일의 확장자를 확인해주는 함수
const checkExtension = (fileName, extensions) => {
    // 파일 이름에서 확장자를 추출
    const fileExtension = fileName.split(".").pop().toLowerCase();
    // 허용할 목록들과 비교
    return extensions.includes(fileExtension);
};

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

// birthYear에 출생연도 1910 ~ 2010 option 생성
const birthYear = document.querySelector(".birth-year");

for (let i = 0; 100 >= i; i++) {
    birthYear.innerHTML += `<option value="${2010 - i}">${2010 - i}</option>`;
}

//저장하기 버튼 클릭 시 모달 나타나는 이벤트
const saveModalWrap = document.querySelector(".save-modal-wrap");
const saveBtn = document.querySelector(".save-btn");

saveBtn.addEventListener("click", () => {
    saveModalWrap.style.display = "block";
});

// 저장하기 모달 내의 버튼 클릭 시 모달창 끄는 이벤트
const modalBtns = document.querySelectorAll(".modal-btn-box button");

modalBtns.forEach((modalBtn) => {
    modalBtn.addEventListener("click", () => {
        saveModalWrap.style.display = "none";
    });
});


// 전화번호 정규식
const hidden = document.getElementById("phone-error");
const phoneInput = document.querySelector(".phone-input");

const autoHyphenAndValidate = (target) => {
    const cleanedValue = target.value.replace(/[^0-9]/g, '');
    target.value = cleanedValue
        .replace(/^(\d{0,3})(\d{0,4})(\d{0,4})$/g, "$1-$2-$3").replace(/(\-{1,2})$/g, "");

    if (cleanedValue.length === 11) {
        hidden.style.display = 'none';
    } else {
        hidden.style.display = 'block';
    }
}

// 초기 로딩 시에도 검사 수행
autoHyphenAndValidate(phoneInput);

phoneInput.addEventListener('input', () => {
    autoHyphenAndValidate(phoneInput);
});

