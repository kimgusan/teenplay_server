const activityLikeCountService = (() => {
    const getCount = async (activityId, callback) => {
        const response = await fetch(`/activity/likes/api?id=${activityId}`);
        const likeCount = await response.json();
        if (callback) {
            callback(likeCount);
        }
    }

    const addOrDeleteLike = async (activityId, isCreate) => {
        await fetch(`/activity/like?id=${activityId}&is-create=${isCreate}`);
    }

    return {getCount: getCount, addOrDeleteLike: addOrDeleteLike}
})()