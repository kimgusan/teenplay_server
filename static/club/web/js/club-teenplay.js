const teenplayTabButton = document.querySelector('.club-detail-filter-teenplay>.club-detail-filter-button')
const teenplayViewList = document.querySelector('.club-teenplay-wrap')
const teenplayMoreButtonWrap = document.querySelector('.tp-show-more-btn-wrap')
const teenplayWrap= document.querySelector('.club-teenplay-contents-box')
const teenplayAddButton = document.querySelector('.tp-show-more-btn')
const resetTeenplayForm = document.getElementById('teenplay-modal-reset-form')
let pageAdd=1
// 시간 변환 함수
function timeSince(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const seconds = Math.floor((now - date) / 1000);

    let interval = seconds / 31536000; // 연 단위

    if (interval > 1) {
        return Math.floor(interval) + "년 전";
    }
    interval = seconds / 2592000; // 월 단위
    if (interval > 1) {
        return Math.floor(interval) + "개월 전";
    }
    interval = seconds / 86400; // 일 단위
    if (interval > 1) {
        return Math.floor(interval) + "일 전";
    }
    interval = seconds / 3600; // 시간 단위
    if (interval > 1) {
        return Math.floor(interval) + "시간 전";
    }
    interval = seconds / 60; // 분 단위
    if (interval > 1) {
        return Math.floor(interval) + "분 전";
    }
    return "방금 전";
}

// 좋아요 변환 함수
function formattingLikeCount(likes){
    if(likes < 1000){
        return likes.toString();
    }
    else if(likes < 10000){
        return (likes/1000).toFixed(1) + 'k';
    }else if(likes < 100000){
        return (likes/10000).toFixed(1) + 'M'
    }else{
        return (likes / 100000).tofixed(1) + '+'
    }
}


// 틴플레이 삭제 함수
function deleteTeenplayAction (teenplayDeleteIdx, clubId){
    Swal.fire({
        title: "삭제하시겠습니까?",
        text: "삭제된 틴플레이는 복구가 불가능합니다.",
        icon: "warning",
        showCancelButton: true,
        confirmButtonColor: "#3085d6",
        cancelButtonColor: "#d33",
        confirmButtonText: "삭제",
        cancelButtonText: "취소",
    }).then((result) => {
        if (result.value) {
            Swal.fire("삭제 완료", "틴플레이 삭제가 완료되었습니다.", "success");
            clubDetailService.updateTeenplayStatus(teenplayDeleteIdx).then(() => {
                clubDetailService.getTeenplayList(clubId, page).then((teenplayInfo) => {
                    teenplayWrap.innerHTML = ''
                    pageAdd =1
                    let teenplayFirstInfo = teenplayInfo.teenplay_list
                    let containTeenplay = teenplayWrap.getElementsByClassName('club-teenplay-contents').length
                    if (teenplayFirstInfo.length > 0){
                        if(containTeenplay === 0){
                            teenplayWrap.innerHTML += showList(teenplayFirstInfo)
                            if(teenplayInfo.has_next){
                                teenplayMoreButtonWrap.style.display='flex'
                            }
                            else{
                                teenplayMoreButtonWrap.style.display = "none"
                            }
                        }
                    }
                    else{
                        teenplayMoreButtonWrap.style.display = "none"
                        teenplayViewList.innerHTML = noneText
                    }
                })
            })
        } else if ((result.dismiss = "cancel")) {
            return;
        }
    });
}



// 틴플레이 없을 때 html
const noneText = `
<div class="club-teenplay-empty-wrap">
    <div class="club-teenplay-empty-container">
        <div class="club-teenplay-empty">업로드한 틴플레이가 없습니다.</div>
    </div>
</div>
`

const showList = (teenplayFirstInfo) => {
    let text =``
    teenplayFirstInfo.forEach((teenplay)=> {
        let teenplayCreateDate = timeSince(teenplay.created_date)
        let teenplayLikeCount = formattingLikeCount(teenplay.like_count)
        if(memberId === teenplay.club__member_id){
            text += `
                <div class="club-teenplay-contents">
                    <a href="/teenplay/club/select/${teenplay.id}" class="club-teenplay-thumbnail-wrap">
                        <img src="/upload/${teenplay.thumbnail_path}" class="club-teenplay-thumbnail" />
                    </a>
                    <div class="club-teenplay-details">
                        <h3 class="club-teenplay-title-wrap">
                            <a href="" class="club-teenplay-title-link">
                                <span class="club-teenplay-title">${teenplay.teenplay_title}</span>
                            </a>
                        </h3>
                        <div class="club-teenplay-views-wrap">
                            <div class="club-teenplay-views-container">
                                <?xml version="1.0" ?><!DOCTYPE svg PUBLIC '-//W3C//DTD SVG 1.1//EN' 'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd'>
                                <svg width="15px" height="15px" fill="currentColor" id="Layer_1" style="enable-background: new 0 0 64 64" version="1.1" viewBox="0 0 64 64" xml:space="preserve" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
                                    <g>
                                        <g id="Icon-Heart" transform="translate(178.000000, 230.000000)">
                                            <path class="st0" d="M-146-177.1l-0.8-0.7c-18.2-14.8-23.1-20-23.1-28.5c0-7,5.7-12.6,12.6-12.6     c5.8,0,9.1,3.3,11.3,5.8c2.2-2.5,5.5-5.8,11.3-5.8c7,0,12.6,5.7,12.6,12.6c0,8.5-4.9,13.7-23.1,28.5L-146-177.1L-146-177.1z      M-157.3-216.3c-5.5,0-10,4.5-10,10c0,7.3,4.6,12.1,21.3,25.8c16.7-13.7,21.3-18.5,21.3-25.8c0-5.5-4.5-10-10-10     c-5,0-7.7,3-9.8,5.4l-1.5,1.7l-1.5-1.7C-149.6-213.3-152.3-216.3-157.3-216.3L-157.3-216.3z" id="Fill-18" />
                                        </g>
                                    </g>
                                </svg>
                                <span class="club-teenplay-views">${teenplayLikeCount}</span>
                            </div>
                            <div class="club-teenplay-date-container">${teenplayCreateDate}</div>
                        </div>
                    </div>
                    <div class="club-teenplay-delete-wrap">
                        <div class="club-teenplay-delete-icon">
                            <img src="/static/club/web/images/club-create/cover_delete_icon_white.png" class="club-teenplay-delete ${teenplay.id}" />
                            <img src="/static/club/web/images/club-detail/teenplay_delete_icon_hover.png" class="club-teenplay-delete-hover ${teenplay.id}" />
                        </div>
                    </div>
                </div>
            `
        }else{
            text += `
                <div class="club-teenplay-contents">
                    <a href="/teenplay/club/select/${teenplay.id}" class="club-teenplay-thumbnail-wrap">
                        <img src="/upload/${teenplay.thumbnail_path}" class="club-teenplay-thumbnail" />
                    </a>
                    <div class="club-teenplay-details">
                        <h3 class="club-teenplay-title-wrap">
                            <a href="" class="club-teenplay-title-link">
                                <span class="club-teenplay-title">${teenplay.teenplay_title}</span>
                            </a>
                        </h3>
                        <div class="club-teenplay-views-wrap">
                            <div class="club-teenplay-views-container">
                                <?xml version="1.0" ?><!DOCTYPE svg PUBLIC '-//W3C//DTD SVG 1.1//EN' 'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd'>
                                <svg width="15px" height="15px" fill="currentColor" id="Layer_1" style="enable-background: new 0 0 64 64" version="1.1" viewBox="0 0 64 64" xml:space="preserve" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
                                    <g>
                                        <g id="Icon-Heart" transform="translate(178.000000, 230.000000)">
                                            <path class="st0" d="M-146-177.1l-0.8-0.7c-18.2-14.8-23.1-20-23.1-28.5c0-7,5.7-12.6,12.6-12.6     c5.8,0,9.1,3.3,11.3,5.8c2.2-2.5,5.5-5.8,11.3-5.8c7,0,12.6,5.7,12.6,12.6c0,8.5-4.9,13.7-23.1,28.5L-146-177.1L-146-177.1z      M-157.3-216.3c-5.5,0-10,4.5-10,10c0,7.3,4.6,12.1,21.3,25.8c16.7-13.7,21.3-18.5,21.3-25.8c0-5.5-4.5-10-10-10     c-5,0-7.7,3-9.8,5.4l-1.5,1.7l-1.5-1.7C-149.6-213.3-152.3-216.3-157.3-216.3L-157.3-216.3z" id="Fill-18" />
                                        </g>
                                    </g>
                                </svg>
                                <span class="club-teenplay-views">${teenplayLikeCount}</span>
                            </div>
                            <div class="club-teenplay-date-container">${teenplayCreateDate}</div>
                        </div>
                    </div>
                    <div class="club-teenplay-delete-wrap hidden">
                        <div class="club-teenplay-delete-icon">
                            <img src="/static/club/web/images/club-create/cover_delete_icon_white.png" class="club-teenplay-delete ${teenplay.id}" />
                            <img src="/static/club/web/images/club-detail/teenplay_delete_icon_hover.png" class="club-teenplay-delete-hover ${teenplay.id}" />
                        </div>
                    </div>
                </div>
            `
        }

    })
    return text
}

const addList = (teenplayAddList) => {
    let text = ``
    let teenplayAddInfo = teenplayAddList.teenplay_list
    if (!teenplayAddList.has_next){
        teenplayMoreButtonWrap.style.display = "none"
    }
    teenplayAddInfo.forEach((teenplay)=> {
        let teenplayCreateDate = timeSince(teenplay.created_date)
        let teenplayLikeCount = formattingLikeCount(teenplay.like_count)
        if(memberId === teenplay.club__member_id){
            text += `
                <div class="club-teenplay-contents">
                    <!-- 경로에 선택한 틴플레이 리스트로 전달 되어야 합니다. count 도 넘길 수 있는 방법이 있는지 확인 필요     -->
                    <a href="/teenplay/club/select/${teenplay.id}" class="club-teenplay-thumbnail-wrap">
                        <img src="/upload/${teenplay.thumbnail_path}" class="club-teenplay-thumbnail" />
                    </a>
                    <div class="club-teenplay-details">
                        <h3 class="club-teenplay-title-wrap">
                            <a href="" class="club-teenplay-title-link">
                                <span class="club-teenplay-title">${teenplay.teenplay_title}</span>
                            </a>
                        </h3>
                        <div class="club-teenplay-views-wrap">
                            <div class="club-teenplay-views-container">
                                <?xml version="1.0" ?><!DOCTYPE svg PUBLIC '-//W3C//DTD SVG 1.1//EN' 'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd'>
                                <svg width="15px" height="15px" fill="currentColor" id="Layer_1" style="enable-background: new 0 0 64 64" version="1.1" viewBox="0 0 64 64" xml:space="preserve" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
                                    <g>
                                        <g id="Icon-Heart" transform="translate(178.000000, 230.000000)">
                                            <path class="st0" d="M-146-177.1l-0.8-0.7c-18.2-14.8-23.1-20-23.1-28.5c0-7,5.7-12.6,12.6-12.6     c5.8,0,9.1,3.3,11.3,5.8c2.2-2.5,5.5-5.8,11.3-5.8c7,0,12.6,5.7,12.6,12.6c0,8.5-4.9,13.7-23.1,28.5L-146-177.1L-146-177.1z      M-157.3-216.3c-5.5,0-10,4.5-10,10c0,7.3,4.6,12.1,21.3,25.8c16.7-13.7,21.3-18.5,21.3-25.8c0-5.5-4.5-10-10-10     c-5,0-7.7,3-9.8,5.4l-1.5,1.7l-1.5-1.7C-149.6-213.3-152.3-216.3-157.3-216.3L-157.3-216.3z" id="Fill-18" />
                                        </g>
                                    </g>
                                </svg>
                                <span class="club-teenplay-views">${teenplayLikeCount}</span>
                            </div>
                            <div class="club-teenplay-date-container">${teenplayCreateDate}</div>
                        </div>
                    </div>
                    <div class="club-teenplay-delete-wrap">
                        <div class="club-teenplay-delete-icon">
                            <img src="/static/club/web/images/club-create/cover_delete_icon_white.png" class="club-teenplay-delete ${teenplay.id}" />
                            <img src="/static/club/web/images/club-detail/teenplay_delete_icon_hover.png" class="club-teenplay-delete-hover ${teenplay.id}" />
                        </div>
                    </div>
                </div>
            `
        }else{
            text += `
                <div class="club-teenplay-contents">
                    <!-- 경로에 선택한 틴플레이 리스트로 전달 되어야 합니다. count 도 넘길 수 있는 방법이 있는지 확인 필요     -->
                    <a href="/teenplay/club/select/${teenplay.id}" class="club-teenplay-thumbnail-wrap">
                        <img src="/upload/${teenplay.thumbnail_path}" class="club-teenplay-thumbnail" />
                    </a>
                    <div class="club-teenplay-details">
                        <h3 class="club-teenplay-title-wrap">
                            <a href="" class="club-teenplay-title-link">
                                <span class="club-teenplay-title">${teenplay.teenplay_title}</span>
                            </a>
                        </h3>
                        <div class="club-teenplay-views-wrap">
                            <div class="club-teenplay-views-container">
                                <?xml version="1.0" ?><!DOCTYPE svg PUBLIC '-//W3C//DTD SVG 1.1//EN' 'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd'>
                                <svg width="15px" height="15px" fill="currentColor" id="Layer_1" style="enable-background: new 0 0 64 64" version="1.1" viewBox="0 0 64 64" xml:space="preserve" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
                                    <g>
                                        <g id="Icon-Heart" transform="translate(178.000000, 230.000000)">
                                            <path class="st0" d="M-146-177.1l-0.8-0.7c-18.2-14.8-23.1-20-23.1-28.5c0-7,5.7-12.6,12.6-12.6     c5.8,0,9.1,3.3,11.3,5.8c2.2-2.5,5.5-5.8,11.3-5.8c7,0,12.6,5.7,12.6,12.6c0,8.5-4.9,13.7-23.1,28.5L-146-177.1L-146-177.1z      M-157.3-216.3c-5.5,0-10,4.5-10,10c0,7.3,4.6,12.1,21.3,25.8c16.7-13.7,21.3-18.5,21.3-25.8c0-5.5-4.5-10-10-10     c-5,0-7.7,3-9.8,5.4l-1.5,1.7l-1.5-1.7C-149.6-213.3-152.3-216.3-157.3-216.3L-157.3-216.3z" id="Fill-18" />
                                        </g>
                                    </g>
                                </svg>
                                <span class="club-teenplay-views">${teenplayLikeCount}</span>
                            </div>
                            <div class="club-teenplay-date-container">${teenplayCreateDate}</div>
                        </div>
                    </div>
                    <div class="club-teenplay-delete-wrap hidden">
                        <div class="club-teenplay-delete-icon">
                            <img src="/static/club/web/images/club-create/cover_delete_icon_white.png" class="club-teenplay-delete ${teenplay.id}" />
                            <img src="/static/club/web/images/club-detail/teenplay_delete_icon_hover.png" class="club-teenplay-delete-hover ${teenplay.id}" />
                        </div>
                    </div>
                </div>
            `
        }

    })
    return text
}


// 틴플레이 처음 클릭했을 떄 나타나는 틴플레이 리스트
teenplayTabButton.addEventListener('click',  (e)=> {
    clubId = club_list[0]['id']
    let page= 1
    clubDetailService.getTeenplayList(clubId, page).then((teenplayInfo) => {
        let teenplayFirstInfo =teenplayInfo.teenplay_list
        if (teenplayFirstInfo.length > 0){
            teenplayWrap.innerHTML = ''
            teenplayWrap.innerHTML += showList(teenplayFirstInfo)
            if(teenplayInfo.has_next){
                teenplayMoreButtonWrap.style.display='flex'
            }
            else{
                teenplayMoreButtonWrap.style.display = "none"
            }
        }
        else{
            teenplayMoreButtonWrap.style.display = "none"
            teenplayViewList.innerHTML = noneText
        }
    })
})

// 더보기 버튼 클릭
teenplayMoreButtonWrap.addEventListener('click', async (e) => {
    const teenplayAddList = await clubDetailService.getTeenplayList(clubId,++pageAdd)
    teenplayWrap.innerHTML += addList(teenplayAddList)
})



// 틴플레이 삭제 선택
teenplayWrap.addEventListener('click', (e) => {
    clubId = club_list[0]['id']
    let deleteIcon = e.target.closest('.club-teenplay-delete-icon')
    if(deleteIcon){
        const teenplayDeleteIdx= deleteIcon.querySelector('img').classList[1];
        deleteTeenplayAction(teenplayDeleteIdx, clubId)
        pageadd = 1
    }
})

// mouseover 이벤트 위임
teenplayWrap.addEventListener('mouseover', (e) => {
    let deleteIcon = e.target.closest('.club-teenplay-delete-icon');
    if (deleteIcon) {
        deleteIcon.querySelector('.club-teenplay-delete').style.display = "none";
        deleteIcon.querySelector('.club-teenplay-delete-hover').style.display = "block";
    }
});

// mouseout 이벤트 위임
teenplayWrap.addEventListener('mouseout', (e) => {
    let deleteIcon = e.target.closest('.club-teenplay-delete-icon');
    if (deleteIcon) {
        deleteIcon.querySelector('.club-teenplay-delete').style.display = "block";
        deleteIcon.querySelector('.club-teenplay-delete-hover').style.display = "none";
    }
});

// 업로드 버튼 클릭 시 업로드 진행
finalSaveButton.addEventListener("click", async () => {
    clubId = club_list[0]['id']
    const teenplayTitleInput = document.querySelector('.name-form-input').value;
    const videoFileInput = document.getElementById('Filedata').files[0]; // 파일을 가져옵니다.
    const thumbnailImageInput = document.getElementById('background-image').files[0]; // 파일을 가져옵니다.

    Swal.fire({
        title: "업로드하시겠습니까?",
        text: "한 번 업로드한 틴플레이는 수정이 불가능합니다.",
        icon: "warning",
        showCancelButton: true,
        confirmButtonColor: "#3085d6",
        cancelButtonColor: "#d33",
        confirmButtonText: "업로드",
        cancelButtonText: "취소",
    }).then(async (result) => {
        if (result.value) {
            // 틴플레이 업로드 관련 서버 작업 코드 입력
            // 여기서 fetch 를 사용해서 진행 되어야 할 것으로 예상
            const formData = new FormData();
            formData.append('title',teenplayTitleInput)
            formData.append('video', videoFileInput)
            formData.append('thumbnail', thumbnailImageInput)
            formData.append('clubId', clubId)
            await clubDetailService.uploadTeenplay(formData)
            tpModalCloseBtn.click();
            clubDetailService.getTeenplayList(clubId, page).then((teenplayInfo) => {
                teenplayWrap.innerHTML = ''
                // 틴플레이 화면 초기화
                let page = 1
                let teenplayFirstInfo = teenplayInfo.teenplay_list
                let containTeenplay = teenplayWrap.getElementsByClassName('club-teenplay-contents').length

                if (teenplayFirstInfo.length > 0){
                    if(containTeenplay === 0){
                        teenplayWrap.innerHTML += showList(teenplayFirstInfo)
                        if(teenplayInfo.has_next){
                            teenplayMoreButtonWrap.style.display='flex'
                        }
                        else{
                            teenplayMoreButtonWrap.style.display = "none"
                        }
                    }
                }
                else{
                    teenplayMoreButtonWrap.style.display = "none"
                    teenplayViewList.innerHTML = noneText
                }
            })
            Swal.fire("업로드 진행중", "업로드를 진행합니다. <br> 업로드는 최대 5분 안에 완료됩니다!", "success");
        } else if (result.dismiss == "cancel") {
            return;
        }
    });
});


// 틴플레이 업로드 리셋 함수
function teenplayResetFunction() {
    resetTeenplayForm.reset()
    tpUploadModal.style.display = "block";
    thumbnailInput.value = "";
    uploadedThumbnailInfo.classList.remove("appear");
    uploadedThumbnailInfo.classList.add("disappear");
    setTimeout(() => {
        uploadedThumbnailInfo.classList.add("hidden");
        thumbnailUploadBox.classList.remove("hidden");
        thumbnailUploadBox.classList.remove("disappear");
        thumbnailUploadBox.classList.add("appear");
    }, 501);
    thumbnailSizeInfo.innerText = "";
    thumbnailNameInfo.innerText = "";

    fileInput.value = "";
    afterUploadModal.classList.remove("appear");
    afterUploadModal.classList.add("disappear");
    setTimeout(() => {
        afterUploadModal.classList.add("hidden");
        beforeUploadModal.classList.remove("hidden");
        beforeUploadModal.classList.remove("disappear");
        beforeUploadModal.classList.add("appear");
    }, 501);
    fileSizeInfo.innerText = "";
    fileNameInfo.innerText = "";

    if (!finalSaveButton.classList.contains("disabled")) {
        finalSaveButton.classList.add("disabled");
    }

    if (teenPlayTextInput.value.length >= 10 && thumbnailInput.files.length) {
        finalSaveButton.classList.remove("disabled");
    } else {
        if (!finalSaveButton.classList.contains("disabled")) {
            finalSaveButton.classList.add("disabled");
        }
    }

    if (teenPlayTextInput.value.length >= 10) {
        finalSaveButton.classList.remove("disabled");
    } else {
        if (!finalSaveButton.classList.contains("disabled")) {
            finalSaveButton.classList.add("disabled");
        }
    }
}

const tpUploadBefore = document.getElementsByName('teenplay-upload-before')
const tpUploadAfter = document.getElementsByName('teenplay-upload-after')
tpUploadBtn.addEventListener('click', () =>{
    teenplayResetFunction()
})