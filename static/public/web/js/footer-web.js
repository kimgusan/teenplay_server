// 푸터에 공지사항 최신 1개 가져오기
const getNotice = (callback) => {
    fetch(`/footer/notice/api/`)
        .then((response) => response.json())
        .then((notice_info) => {
            if (callback) {
                callback(notice_info)
            }
        })
}

const showNotice = (notice_info) => {
    let notices = notice_info.notices;
    notices.forEach((notice) => {
        const noticeTitle = document.querySelector(".new-notice-title");
        noticeTitle.innerText = notice.notice_title;
    })
}

getNotice(showNotice);