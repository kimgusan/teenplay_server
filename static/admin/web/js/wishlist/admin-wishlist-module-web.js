const adminWishlistService = (() => {
    // 페이지 가져오기
    const getPagination = async (page, category, type, keyword, wishlist_id, callback) => {
        const response = await fetch(`/admin/wishlists/${page}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({
                'page': page,
                'category': category,
                'type': type,
                'keyword': keyword,
                'wishlist_id': wishlist_id
            })
        })
        const pagination = await response.json()

        if(callback) {
            return  callback(pagination)
        }
        return pagination;
    }

    // 위시리스트 삭제
    const remove = async (targetId) => {
        const wishlist_id = targetId.targetId

        await fetch(`/admin/wishlists/delete/${wishlist_id}/`, {
            method: 'delete',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'wishlist_id': wishlist_id})
        });
    }

    return {getPagination: getPagination, remove: remove}
})();