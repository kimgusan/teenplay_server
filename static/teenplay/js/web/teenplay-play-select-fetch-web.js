const teenplayClubService = (() => {
    const getList = async (clubId, page, teenplayClickId, callback) => {
        let url = `/teenplay/club/select/api/${clubId}/`

        if(page !== undefined && page!= null){
            url += `${page}/`
        }

        url+= `${teenplayClickId}`

        const response = await fetch(url);
        const teenplayList = await response.json();
        if(callback){
            return callback(teenplayList)
        }
        return teenplayList
    }

    const likeTeenplay = async (teenplayId, memberSessionId, displayStyle, callback) => {
        const teenplayLikeResponse = await fetch(`/teenplay/club/select/like/api/${teenplayId}/${memberSessionId}/${displayStyle}/`);
        const videoLike = await teenplayLikeResponse.json();
        if (callback) {
            return callback(videoLike);
        }
        return(videoLike)
    };

    return {
        getList: getList,
        likeTeenplay:likeTeenplay
    }
})()