const adminMessageService = (() => {
    // 페이지 데이터 불러오기
    const getPagination = async (page, category, type, keyword, callback) => {
        const response = await fetch(`/admin/messages/${page}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({
                'page': page,
                'category': category,
                'type': type,
                'keyword': keyword
            })
        })
        const pagination = await response.json()

        if(callback) {
            return  callback(pagination)
        }
        return pagination;
    }


    // 회원 상태 변경
    const update = async (targetId) => {
        const member_id = targetId.targetId

        await fetch(`/admin/user/update/${member_id}/`, {
            method: 'patch',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'member_id': member_id})
        });

    }


    return {getPagination: getPagination, update: update}
})();
