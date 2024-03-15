const adminMessageService = (() => {
    // 페이지 데이터 불러오기
    const getPagination = async (page, category, type, keyword, callback) => {
        const response = await fetch(`/admin/messages/api/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({
                page: page,
                category: category,
                type: type,
                keyword: keyword
            })
        })
        const pagination = await response.json()

        if(callback) {
            return  callback(pagination)
        }
        return pagination;
    }


    // 쪽지 삭제
    const remove = async (page, targetId) => {
        const message_id = targetId.targetId

        await fetch(`/admin/messages/${page}?message_id=${message_id}`, {
            method: 'fetch',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'message_id': message_id})
        });
    }


    return { getPagination: getPagination, remove: remove}
})();