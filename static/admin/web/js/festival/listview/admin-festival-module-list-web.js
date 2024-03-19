// 공지사항 관리자 서비스 생성
const adminFestivalService = (() => {
    const write = async (festival) => {
        const response= await fetch(`/admin/festival/write/`, {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify(festival)
        });
    }

    const getPagination = async (page, callback) => {
        const response = await fetch(`/admin/festivals/${page}/`);
        const pagination = await response.json();

        if (callback){
            return callback(pagination);
        }
        return pagination;
    }

    // const getCategory = async (page, category, callback) => {
    //     const response = await fetch(`/admin/festivals/${page}?category=${category}`);
    //     const pagination = await response.json();
    //
    //     if (callback){
    //         return callback(pagination);
    //     }
    //     return pagination;
    // }


    // 축제 삭제
    const remove = async (targetId) => {
        const festival_id = targetId.targetId

        await fetch(`/admin/festivals/delete/${festival_id}/`, {

            method: 'patch',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'festival_id': festival_id})
        });
        console.log("festival_id 들어옴")
        console.log(festival_id)
    }

    // 상세보기 가져오기
    const showDetail = async (page, targetId, callback) => {
        console.log(targetId)

        const response = await fetch(`/admin/festivals/${page}?targetId=${targetId}`);
        const pagination = await response.json();

        if (callback){
            return callback(pagination);
        }
        return pagination;
    }

    // 검색하기
    const search = async (page, keyword, callback) => {

        const response = await fetch(`/admin/festivals/${page}/?keyword=${keyword}`)
        const pagination = await response.json();

        if (callback){
            return callback(pagination);
        }

        return pagination;
    }

    return {write:write, getPagination:getPagination, remove: remove, showDetail:showDetail, search:search}
})();
