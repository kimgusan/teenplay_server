const adminMeetingModule = (() => {
    const getPagination = async (page, keyword, callback) => {
        const response = await fetch(`/admin/meetings/${page}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({
                'page': page,
                'keyword': keyword
            })
        })
        const pagination = await response.json();

        if (callback){
            return callback(pagination);
        }
        return pagination;
    }

    const remove = async (targetId) => {
        const meeting_id = targetId.targetId

        const response = await fetch(`/admin/meetings/delete/${meeting_id}`, {
            method: 'delete',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'meeting_id': meeting_id})
        });
    }

    return {getPagination:getPagination, remove:remove}
})();