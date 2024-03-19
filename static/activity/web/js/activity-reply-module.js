const activityReplyService = (() => {
    const write = async (activityReply) => {
        const response = await fetch(`/activity/replies/api/`, {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify(activityReply)
        });
    };

    const getList = async (isAdd, page, activityId, callback) => {
        const response = await fetch(`/activity/replies/api/?page=${page}&activity-id=${activityId}`);
        const replies = await response.json();
        if (callback) {
            callback(isAdd, replies);
        }
    }

    const update = async (activityReply) => {
        await fetch(`/activity/replies/api/`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify(activityReply)
        });
    }

    const remove = async (id) => {
        await fetch(`/activity/replies/api/`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify(id)
        });
    }

    return {write: write, getList: getList, update: update, remove: remove}
})()