// 공지사항 관리자 서비스 생성
const adminNoticeService = (() => {
    const write = async (notice_info) => {
        await fetch(`/admin/notice/write/`, {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify(notice_info)
        });
    }

    // 페이지 가져오기
    const getPagination = async (page, category, keyword, callback) => {
        const response = await fetch(`/admin/notices/${page}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({
                'page': page,
                'category': category,
                'keyword': keyword
            })
        })
        const pagination = await response.json()

        if(callback) {
            return  callback(pagination)
        }
        return pagination;
    }


    // 공지사항 수정
    const update = async (targetID, titleValue, contentValue) => {
        const notice_id = targetID.targetID
        const title = titleValue.titleValue
        const content = contentValue.contentValue

        await fetch(`/admin/notices/update/${notice_id}/`, {
            method: 'post',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'notice_id': notice_id, title: title, content: content})
        });

    }


    // 공지사항 삭제
    const remove = async (targetId) => {
        const notice_id = targetId.targetId

        await fetch(`/admin/notices/update/${notice_id}/`, {
            method: 'delete',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'notice_id': notice_id})
        });
    }

    return {write:write, getPagination:getPagination, update:update, remove: remove}
})();
