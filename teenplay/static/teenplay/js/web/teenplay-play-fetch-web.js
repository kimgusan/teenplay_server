const teenplayMainService = (() => {
    const getTeenplay = async (slideNumber, callback) => {
        const response = await fetch(`new/api/${slideNumber}`)
        const teenplay = await response.json();
        if (callback){
            callback(teenplay)
        }
        return teenplay
    }

    const likeTeenplay = async (teenplayId, memberSessionId, displayStyle, callback) => {
        const teenplayLikeResponse = await fetch(`/like/api/${teenplayId}/${memberSessionId}/${displayStyle}/`);
        console.log(teenplayId)
        console.log(memberSessionId)
        console.log(displayStyle)
        const videoLike = await teenplayLikeResponse.json();
        if (callback) {
            return callback(videoLike);
        }
        return(videoLike)
    };

    return {
        getTeenplay: getTeenplay,
        likeTeenplay: likeTeenplay
    }
})()