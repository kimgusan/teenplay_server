const postLists = document.querySelectorAll(".teenchin-list-add");

postLists.forEach((postList) => {
    const postListScrollWidth = postList.scrollWidth;
    const postListClientWidth = postList.clientWidth;

    let startX = 0;
    let nowX = 0;
    let endX = 0;
    let listX = 0;

    const onScrollStart = (e) => {
        startX = getClientX(e);
        window.addEventListener("mousemove", onScrollMove);
        window.addEventListener("touchmove", onScrollMove);
        window.addEventListener("mouseup", onScrollEnd);
        window.addEventListener("touchend", onScrollEnd);
    };

    const onScrollMove = (e) => {
        nowX = getClientX(e);
        setTranslateX(listX + nowX - startX);
    };

    const onScrollEnd = (e) => {
        endX = getClientX(e);
        listX = getTranslateX();
        if (listX > 0) {
            setTranslateX(0);
            postList.style.transition = `all 0.3s ease`;
            listX = 0;
        } else if (listX < postListClientWidth - postListScrollWidth) {
            setTranslateX(postListClientWidth - postListScrollWidth);
            postList.style.transition = `all 0.3s ease`;
            listX = postListClientWidth - postListScrollWidth;
        }

        window.removeEventListener("mousedown", onScrollStart);
        window.removeEventListener("touchstart", onScrollStart);
        window.removeEventListener("mousemove", onScrollMove);
        window.removeEventListener("touchmove", onScrollMove);
        window.removeEventListener("mouseup", onScrollEnd);
        window.removeEventListener("touchend", onScrollEnd);
        window.removeEventListener("click", onClick);

        setTimeout(() => {
            bindEvents();
            postList.style.transition = "";
        }, 300);
    };

    const onClick = (e) => {
        if (startX - endX !== 0) {
            e.preventDefault();
        }
    };

    const getClientX = (e) => {
        const isTouches = e.touches ? true : false;
        return isTouches ? e.touches[0].clientX : e.clientX;
    };

    const getTranslateX = () => {
        return parseInt(getComputedStyle(postList).transform.split(/[^\-0-9]+/g)[5]);
    };

    const setTranslateX = (x) => {
        postList.style.transform = `translateX(${x}px)`;
    };

    const bindEvents = () => {
        postList.addEventListener("mousedown", onScrollStart);
        postList.addEventListener("touchstart", onScrollStart, { passive: true });
        postList.addEventListener("click", onClick);
    };

    bindEvents();
});
