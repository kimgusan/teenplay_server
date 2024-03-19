const clubPostRelyService = (() => {
    const write = async (clubPostReply) => {
        const response = await fetch("/clubs/pr-post-reply/api/", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify(clubPostReply)
        });
    }

    const getList = async (clubPostId, page, callback) => {
        const response = await fetch(`/clubs/pr-post-reply/api/?club_post_id=${clubPostId}&page=${page}`);
        const replies_info = await response.json();
        if (callback){
            return callback(replies_info);
        }

        return replies_info;
    }

    const update = async (clubPostReply) => {
        await fetch(`/clubs/pr-post-reply/api/`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify(clubPostReply)
        });
    }

    const remove = async (id) => {
        await fetch(`/clubs/pr-post-reply/api/?id=${id}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            }
        })
    }

    return {write: write, getList: getList, update: update, remove: remove}
})()