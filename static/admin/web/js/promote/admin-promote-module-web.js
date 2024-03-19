const adminPromoteService = (() => {
    // 페이지 가져오기
    const getPagination = async (page, type, keyword, callback) => {
        const response = await fetch(`/admin/promotes/${page}/`, {
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

    // 게시글 삭제
    const remove = async (targetId) => {
        const promote_id = targetId.targetId

        await fetch(`/admin/promotes/delete/${promote_id}/`, {
            method: 'delete',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'promote_id': promote_id})
        });
    }

    return {getPagination: getPagination, remove: remove}
})();