const adminActivityService = (() => {
    // 페이지 가져오기
    const getPagination = async (page, type, keyword, callback) => {
        const response = await fetch(`/admin/activities/${page}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({
                'page': page,
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

    // 위시리스트 삭제
    const remove = async (targetId) => {
        const activity_id = targetId.targetId

        await fetch(`/admin/activities/delete/${activity_id}/`, {
            method: 'delete',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'activity_id': activity_id})
        });
    }

    return {getPagination: getPagination, remove: remove}
})();