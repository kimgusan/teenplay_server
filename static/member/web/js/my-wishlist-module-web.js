const wishlistService = (() => {

    const getList = async (member_id, page, status_wishlist, callback) => {
        const response = await fetch(`/member/mypage-wishlist/${member_id}/${page}/?status_wishlist=${status_wishlist}`);
        const response_data = await response.json();
        // 서버 응답에서 전체 페이지 수 추출

        // 각각의 데이터에 접근
        const wishlists = response_data.wishlists;
        const total_pages = response_data.total_pages;
        // callback 함수에 필요한 데이터 전달
        return callback(wishlists,total_pages);
}


    const remove = async (wishlist_id) => {
        await fetch(`/member/mypage-wishlist/${wishlist_id}/`, {
            method: 'delete',
            headers: {'X-CSRFToken': csrf_Token}
        });
    }


    return {getList:getList,remove:remove}
})();