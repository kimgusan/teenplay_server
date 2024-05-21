const clubPostListService = (() => {
    const getList = async (page, category, order, keyword, callback) => {
        const response = await fetch(`/clubs/pr-post-list/api/?page=${page}&category=${category}&order=${order}&keyword=${keyword}`)
        const clubPostList = await response.json()

        if(callback) {
            return  callback(clubPostList)
        }
        return clubPostList
    }

    return {getList: getList}
})();
