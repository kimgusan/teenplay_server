let page = 1;

const div = document.querySelector(".add-post");
const addButton = document.querySelector(".post-watch");
const moreButton = document.querySelector(".post-watch-button")
const categoryBtn = document.querySelectorAll(".top-categroy-item")
let myWishlistIdCheck = myWishlistId ? myWishlistId : 0;



const showList = (data) => {
    let text = ``;
    if(data['wishlists'].length === 0) {
        text = `
            <!-- 카테고리에  게시글이 없을 때 출력 -->
            <div class="post-error" style="text-align: center;">
                <span>게시글이 존재하지 않습니다</span>
            </div>
        `
    }
    else {
        data['wishlists'].forEach((wishlist)=> {
            text += `
                <!-- 위시리스트 게시글 부분 -->
                <div class="wishlist-post ${data['my_wishlist']}">
                    <div class="post-wrap">
                        <!-- 위시리스트 게시글 상단 부분 -->
                        <div class="post-top-warp ${wishlist.id}">
                            <!-- 위시리스트 게시글 프로필 부분 -->
                            <div class="post-profile-warp ${wishlist.id}">
                            <input type="hidden" class="member-email${wishlist.id}" name="writer-email" value="${wishlist.member_email}">
                                <!-- 위시리스트 게시글 프로필 이미지 부분 -->
                                <div class="post-profile-img-container ${wishlist.id}">
                                    <img src="${wishlist.member_path ? '/upload/' + wishlist.member_path : '/static/public/web/images/logo/logo1.png'}" alt="프로필사진" class="post-profile-icon profile-image${wishlist.id}">
                                </div>
                                <div class="post-username-container">
                                    <!-- 위시리스트 게시글 프로필 이름 부분 -->
                                    <div class="post-username-text">
                                        <span class="member-name${wishlist.id}" id="${wishlist.is_private}">${wishlist.member_name}</span>
                                    </div>
                                    <!-- 위시리스트 작성자 id -->
                                    <input type="hidden" name="writer-id" class="wishlist-writer-id${wishlist.id}" value="${wishlist.member_id}">
                                    <!-- 위시리스트 게시글 정보 부분 -->
                                    <div class="wishlist-post-info-container">
                                        <span class="post-category${wishlist.id}" id="${wishlist.category_id}">${wishlist.category_name}</span>
                                        <span class="post-upload-date">${timeForToday(wishlist.created_date)}</span>
                                    </div>
                                </div>
                            </div>
                            <!-- 위시리스트 게시글 아이템 부분 -->
                            <div class="post-items">
                                <div class="post-items-wrap">
                                    <div class="post-items-divison-wrap">
                                        <!-- 위시리스트 게시글 좋아요 정보 부분 -->
            `
            // console.log(data['likes'])
            console.log(wishlist.id)
        // for (let i = 0; i < data['wishlists'].length; i++) {}
            if (wishlist.like_on === 1) {
                text += `
                                        <div class="post-like-container ${wishlist.id}">
                                            <svg class="post-like-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" style="display:none" >
                                                <path d="M9.653 16.915l-.005-.003-.019-.01a20.759 20.759 0 01-1.162-.682 22.045 22.045 0 01-2.582-1.9C4.045 12.733 2 10.352 2 7.5a4.5 4.5 0 018-2.828A4.5 4.5 0 0118 7.5c0 2.852-2.044 5.233-3.885 6.82a22.049 22.049 0 01-3.744 2.582l-.019.01-.005.003h-.002a.739.739 0 01-.69.001l-.002-.001z"></path>
                                            </svg>
                                            <svg class="post-like-icon liked" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" style="display:block">
                                                <path d="M9.653 16.915l-.005-.003-.019-.01a20.759 20.759 0 01-1.162-.682 22.045 22.045 0 01-2.582-1.9C4.045 12.733 2 10.352 2 7.5a4.5 4.5 0 018-2.828A4.5 4.5 0 0118 7.5c0 2.852-2.044 5.233-3.885 6.82a22.049 22.049 0 01-3.744 2.582l-.019.01-.005.003h-.002a.739.739 0 01-.69.001l-.002-.001z"></path>
                                            </svg>
                                            <span class="post-like-count">${wishlist.like_total}</span>
                                        </div>
                                        `
            } else {
                text += `
                                        <div class="post-like-container ${wishlist.id}">
                                            <svg class="post-like-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" style="display:block" >
                                                <path d="M9.653 16.915l-.005-.003-.019-.01a20.759 20.759 0 01-1.162-.682 22.045 22.045 0 01-2.582-1.9C4.045 12.733 2 10.352 2 7.5a4.5 4.5 0 018-2.828A4.5 4.5 0 0118 7.5c0 2.852-2.044 5.233-3.885 6.82a22.049 22.049 0 01-3.744 2.582l-.019.01-.005.003h-.002a.739.739 0 01-.69.001l-.002-.001z"></path>
                                            </svg>
                                            <svg class="post-like-icon liked" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" style="display:none">
                                                <path d="M9.653 16.915l-.005-.003-.019-.01a20.759 20.759 0 01-1.162-.682 22.045 22.045 0 01-2.582-1.9C4.045 12.733 2 10.352 2 7.5a4.5 4.5 0 018-2.828A4.5 4.5 0 0118 7.5c0 2.852-2.044 5.233-3.885 6.82a22.049 22.049 0 01-3.744 2.582l-.019.01-.005.003h-.002a.739.739 0 01-.69.001l-.002-.001z"></path>
                                            </svg>
                                            <span class="post-like-count">${wishlist.like_total}</span>
                                        </div>
                                        `
            }
                text += `                        
                                        <!-- 위시리스트 게시글 댓글 정보 부분 -->
                                        <div class="post-comment-container">
                                            <svg class="post-comment-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                                                <path fill-rule="evenodd" d="M4.848 2.771A49.144 49.144 0 0112 2.25c2.43 0 4.817.178 7.152.52 1.978.292 3.348 2.024 3.348 3.97v6.02c0 1.946-1.37 3.678-3.348 3.97a48.901 48.901 0 01-3.476.383.39.39 0 00-.297.17l-2.755 4.133a.75.75 0 01-1.248 0l-2.755-4.133a.39.39 0 00-.297-.17 48.9 48.9 0 01-3.476-.384c-1.978-.29-3.348-2.024-3.348-3.97V6.741c0-1.946 1.37-3.68 3.348-3.97z" clip-rule="evenodd"></path>
                                            </svg>
                                            <span class="post-comment-count">${wishlist.reply_total}</span>
                                        </div>
                                    </div>
                                </div>
                                <div>
                `
            if(wishlist.member_id === Number(memberId)) {
                text += `
                                    <!-- 위시리스트 게시글 메뉴 부분 -->
                                    <button class="post-menu-container" id="main-wishlist-content-menu" type="button" aria-haspopup="menu" aria-expanded="false" data-headlessui-state class="text-gray-500 pb-2 pl-2">
                                        <svg class="post-menu-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                                            <path fill-rule="evenodd" d="M10.5 6a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0zm0 6a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0zm0 6a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0z" clip-rule="evenodd"></path>
                                        </svg>
                                    </button>
                                    <!-- 위시리스트 게시글 메뉴 열기 부분 -->
                                    <div class="post-menu-open hidden" aria-labelledby="main-wishlist-content-menu" role="menu" tabindex="0" data-headlessui-state="open">
                                        <div class="post-menu-open-container" role="none">
                                            <div class="post-menu-open-divison" role="none">
                                                <button class="post-menu-open-choice ${wishlist.id}" type="button" id="post-menu-open-update" role="menuitem" tabindex="-1">수정</button>
                                                <button class="post-menu-open-choice ${wishlist.id}" type="button" id="post-menu-open-delete" role="menuitem" tabindex="-1">삭제</button>
                                            </div>
                                        </div>
                                    </div>
                `
            }
            text += `
                                </div>
                            </div>
                        </div>
                        <!-- 위시리스트 게시글 제목 부분 -->
                        <div class="post-title-wrap">
                            <div class="post-title-text${wishlist.id}">${wishlist.wishlist_content}</div>
                        </div>
                            <!-- 위시리스트 게시글 태그 부분 -->
                            <div class="post-tags-wrap ${wishlist.id}">
                `
            for (let i = 0; i < data['tags'][wishlist.id].length; i++) {
                            text += `<span class="post-tag">${data['tags'][wishlist.id][i]}</span>`;
                        }
            text += `
                            </div>
                    </div>
                    <!-- 위시리스트 댓글 열기 부분 -->
                    <div class="comment-open-wrap" id="open${wishlist.id}">
                        <button class="comment-open-button" type="button">
                            <span class="reply-open-button ${wishlist.id}"> 댓글 보기 </span>
                            <svg class="comment-open-button-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                                <path fill-rule="evenodd" d="M12.53 16.28a.75.75 0 01-1.06 0l-7.5-7.5a.75.75 0 011.06-1.06L12 14.69l6.97-6.97a.75.75 0 111.06 1.06l-7.5 7.5z" clip-rule="evenodd"></path>
                            </svg>
                        </button>
                    </div>
                    <!-- 위시리스트 댓글 닫기 부분 -->
                    <div class="comment-close-wrap hidden" id="close${wishlist.id}">
                        <button class="comment-close-button" type="button">
                            <span class="reply-close-button ${wishlist.id}"> 닫기 </span>
                            <svg class="comment-close-button-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clip-rule="evenodd"></path>
                            </svg>
                        </button>
                    </div>
                    <!-- 위시리스트 댓글 부분 -->
<!--                    <div class="comment-all-wrap hidden" id="reply-form${wishlist.id}">-->
                        <div class="comment-all-wrap ${wishlist.id} hidden" id="reply-form${wishlist.id}">
                    </div>
                </div>
            `
        });
    }
    return text;
}


// 위시리스트 게시글 : 프로필 클릭시 프로필 모달 보여주기 함수
const addClickEventWishlistProfile = () => {
    const profilePhotos = document.querySelectorAll(".post-profile-img-container");
    profilePhotos.forEach((wrap) => {
        wrap.addEventListener("click", (e) => {
            let wishlistId = wrap.classList[1];
            showMemberProfileModal(wishlistId);
        })
    })

    const modalDivision = document.querySelector(".modal-divison.teenchin-modal")
    const modalContainer = document.querySelector(".teenchin-box.post-update-wrap")
    modalDivision.addEventListener("click", (e) => {
        if (e.target !== modalContainer){
            hideMemberProfileModal()
        }
    })
}

// 위시리스트 게시글 메뉴 열고 닫기 함수
const menuOpenevents = () => {

    // 위시리스트 게시글 내 메뉴 버튼 클릭 시 수정/삭제 메뉴 나오기
    const wishlistPostMenuButton = document.querySelectorAll(".post-menu-container");
    const wishlistPostMenu = document.querySelectorAll(".post-menu-open");
    wishlistPostMenuButton.forEach((btn, i) => {
        wishlistPostMenuButton[i].addEventListener("click", () => {
            wishlistPostMenu[i].classList.toggle("hidden");
        });

        // 여백 클릭 시 위시리스트 메뉴 닫기
        document.addEventListener("click", (e) => {
            const clickedElement = e.target;
            if (!wishlistPostMenuButton[i].contains(clickedElement) && !wishlistPostMenu[i].contains(clickedElement)) {
                wishlistPostMenu[i].classList.add("hidden");
            }
        });
    })
}

// 위시리스트 게시글 수정시 사용되는 이벤트 함수
const addAllevents = () => {
    // 위시리스트 수정 메뉴 버튼
    let postUpdateButtons = document.querySelectorAll(".post-menu-open-choice")
    const modalPostUpdate = document.querySelector(".post-update");
    const modalPostUpdateClose = document.querySelector(".update-close-container");
    postUpdateButtons.forEach((button) => {
        button.addEventListener("click", () => {
            modalPostUpdate.classList.remove("hidden");
        });
    });

    // 위시리스트 수정 돌아가기 버튼 클릭시 화면 사라지기 이벤트
    modalPostUpdateClose.addEventListener("click", () => {
        modalPostUpdate.classList.add("hidden");
    });

    // 위시리스트 수정 발행하기 버튼 클릭시 화면 사라지기 이벤트
    const modalPostUpdateFinish = document.querySelector(".update-finish-botton");
    modalPostUpdateFinish.addEventListener("click", () => {
        modalPostUpdate.classList.add("hidden");
    });

    // 위시리스트 게시글 수정 타이틀 입력 이벤트
    // 수정 모달에 타이틀이 입력되지 않으면 안내 문구 빨간색으로 출력
    const updatePostTitle = document.querySelector(".update-category-input");
    modalPostUpdateFinish.addEventListener("click", () => {
        if (!updatePostTitle.value) {
            updatePostTitle.classList.add("red");
            return;
        } else {
            updatePostTitle.classList.remove("red");
        }
    });

    // 위시리스트 게시글 태그 수정 이벤트
    tagInput.addEventListener("keyup", (e) => {
        if (e.keyCode === 13) {
            const tags = document.querySelectorAll(".update-tags-wrap .tag-list");

            // 입력 값의 길이가 10 이하이며, 목록의 태그 개수가 5이하면 태그 값 추가(태그 생성)
            if (e.target.value.length <= 10 && tags.length <= 5) {
                if (e.target.value) {
                    modifyWishlistTag(e.target.value);
                }
            } else {
                // 위 값이 아니라면, 오류 메시지 출력
                tagError.classList.remove("hidden")
            }
        }
    });

    // 위시리스트 게시글 수정 모달창 닫기 이벤트 (닫음과 동시에 내용 초기화)
    const postUpdateWrap = document.querySelector(".post-update-wrap");
    document.addEventListener("click", (e) => {
        let postUpdateButtons = document.querySelectorAll(".post-menu-open-choice")
        for (let btn of postUpdateButtons) {
            if (btn.contains(e.target)) {
                return false;
            }
        }
        if (!postUpdateWrap.contains(e.target)) {
            modalPostUpdate.classList.add("hidden");

            const wishlistContent = document.getElementById("update-content")
            const isPrivate = document.getElementById("update-private")
            const category = document.getElementById("update-category")
            const tagElements = document.querySelector('.tag-list');

            wishlistContent.value = "";
            isPrivate.value = "1";
            category.value = "100";
            tagElements.innerHTML = "";
        }
    });

}

// 더보기 버튼 검사하는 함수
const addMoreButton = async () => {
    await wishlistService.getList(0,page+1, category, keyword).then((data) => {
        // console.log(data['wishlists'].length)
        if (data['wishlists'].length !== 0) {
            addButton.style.display = "block";
        } else if (data['wishlists'].length === 0) {
            addButton.style.display = "none";
        }
    })
}

// 위시리스트 페이지 실행시 위시리스트 보여주기
wishlistService.getList(myWishlistIdCheck, page, category, keyword, showList).then(async (text) => {
    div.innerHTML += text;
    await addMoreButton()
    await addClickEventWishlistProfile()
    await addAllevents();
    await menuOpenevents();
    findFirst();
});

// 각 카테고리 버튼 클릭 시 카테고리별 위시리스트 보여주기
categoryBtn.forEach((button) => {
    button.addEventListener('click', async () => {
        page = 1

        // 특정 카테고리 선택 시 전체 카테고리의 선택 속성 해제 후 선택한 카테고리에만 선택 속성 부여 이벤트
        categoryBtn.forEach((item) => {
            item.classList.remove("all");
        });
        button.classList.add("all");

        // 카테고리에 해당하는 위시리스트 게시글 보여주기
        await wishlistService.getList(myWishlistIdCheck, page, category, keyword, showList).then(async (text) => {
            div.innerHTML = text;
            mySearchInput.value ="";
            await addMoreButton()
            await addClickEventWishlistProfile()
            await addAllevents();
            await menuOpenevents();
        });
    });
});


// 태그 검색시 검색 결과 보여주기
mySearchInput.addEventListener("keyup", async (e) => {
    if (e.keyCode === 13) {
        // console.log("들어옴")
        await wishlistService.getList(0, page, category, keyword, showList).then(async (text) => {
            div.innerHTML = text;
            await addMoreButton()
            await addClickEventWishlistProfile()
            await addAllevents();
            await menuOpenevents();
        });
    }
})


// 더보기 버튼 누르면 위시리스트 추가로 나오기
moreButton.addEventListener("click", async (e) => {
    await wishlistService.getList(0, ++page, category,keyword, showList).then(async (text) => {
        div.innerHTML += text;
        await addMoreButton()
        await addClickEventWishlistProfile()
        await addAllevents();
        await menuOpenevents();
    })
})


// 위시리스트 메뉴 열고 닫기 (수정/삭제)
div.addEventListener("click", async (e) => {
    // const postMenuContainer = e.target.closest('.post-menu-icon');
    // if (e.target === postMenuContainer) {
    if (e.target.classList[0] === 'post-menu-post-menu-container') {
        menuOpenevents();
    }
});


const replyshowList = (replies) => {
    // console.log(replies.length)
    let text = ``;
    if (loginCheck) {
        text = `
                            <!-- 위시리스트 댓글 추가 부분 -->
                            <div class="comment-input-box-all-wrap">
                                <div class="comment-input-box-wrap">
                                    <div class="comment-input-wrap">
                                        <div class="comment-input-username-container">${memberName}</div>
                                            <input type="hidden" name="csrfmiddlewaretoken" id="csrfmiddlewaretoken" value="${csrf_token}">
                                            <input class="comment-input-hidden" name="CommunityBoardArticleId" hidden="" />
                                            <textarea class="comment-input-guide" id="reply-content" name="content" type="text" placeholder="댓글을 남겨주세요. 욕설, 비방글은 무통보 삭제됩니다." autocomplete="off" required=""></textarea>
                                            <div class="comment-input-upload-wrap">
                                                <div class="comment-input-upload-container">
                                                    <button class="comment-input-upload-button" id="comment-upload-button" type="submit">등록</button>
                                                </div>
                                            </div>
                                    </div>
                                </div>
                            </div>
                            <!-- 위시리스트 댓글 개수 정보 부분 -->
                            <div class="comment-count-wrap">댓글 ${replies.length}</div>
             `
    }
    replies.forEach((reply) => {
            text += `
                        <!-- 위시리스트 전체 댓글 부분 -->
                        <div class="comment-list-box-wrap">
                            <!-- 댓글 구분선 -->
                        <!-- <div class="comment-line"></div> -->
                            <!-- 위시리스트 댓글 수정 부분 -->
                            <div class="comment-update-box-all-wrap hidden">
                                <div class="comment-update-box-wrap">
                                    <div class="comment-update-wrap">
                                        <div class="comment-update-username-container">${reply.member_name}</div>
                                        <form class="comment-update-container" method="post" data-hs-cf-bound="true">
                                            <input class="comment-update" name="__RequestVerificationToken" type="hidden" value="lQ1k2HcglUXXMwOa0SIMa_JXSkG8dcFibXk3jrgj1xqrlfkZnxzZ69pIbMszXgP7-1CrLPr7j5y-xEsj8UfSWd2EpmvQnlNfBilqDAT5vUaOv7fZgaT00ILqYk8j9LZOYraW8JeO8poPFXuOtKyENA2" />
                                            <input class="comment-update-hidden" name="CommunityBoardArticleId" hidden="" />
                                            <textarea class="comment-update-guide" id="comment-input-update-guide" name="content" type="text" placeholder="수정할 댓글을 입력하세요" autocomplete="off" value="" required=""></textarea>
                                            <div class="comment-update-upload-wrap">
                                                <div class="comment-update-upload-container">
                                                    <button class="comment-update-upload-button" id="comment-update-upload" type="button">등록</button>
                                                </div>
                                            </div>
                                        </form>
                                    </div>
                                </div>
                            </div>
                            <!-- 위시리스트 댓글 목록 부분 -->
                            <div class="comment-list-all-wrap">
                                <!-- 위시리스트 개별 댓글 부분 -->
                                <div class="comment-list-wrap">
                                <input type="hidden" class="member-email${reply.id}" name="writer-email" value="${reply.member_email}">
                                    <!-- 위시리스트 댓글 내 프로필 사진 부분 -->
                                    <div class="comment-profile-container ${reply.id}">
                                        <img src="${reply.member_path ? '/upload/' + reply.member_path : '/static/public/web/images/logo/logo1.png'}" alt="프로필사진" class="comment-profile-icon profile-image${reply.id}">
                                    </div>
                                    <!-- 위시리스트 개별 댓글 전체 내용 부분 -->
                                    <div class="comment-content-container">
                                        <!-- 위시리스트 댓글 정보 부분 -->
                                        <div class="comment-info">
                                            <!-- 위시리스트 작성자 id -->
                                            <input type="hidden" name="writer-id" class="wishlist-writer-id${reply.id}" value="${reply.member_id}">
                                            <!-- 위시리스트 댓글 작성자 이름 부분 -->
                                            <span class="member-name${reply.id}">${reply.member_name}</span>
                                            <!-- 위시리스트 댓글 작성 날짜 부분 -->
                                            <span class="comment-info-date">${timeForToday(reply.created_date)}</span>
                                        </div>
                                        <!-- 위시리스트 개별 댓글 내용 부분 -->
                                        <div class="comment-text${reply.id}">${reply.reply_content}</div>
                                    </div>
                                </div>
            `
            if(reply.member_id === Number(memberId)) {
                text += `      
                                <!-- 위시리스트 개별 댓글 메뉴 부분 -->
                                <div>
                                    <button class="comment-menu" type="button" aria-haspopup="menu" data-headlessui-state>
                                        <svg class="comment-menu-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                                            <path fill-rule="evenodd" d="M10.5 6a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0zm0 6a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0zm0 6a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0z" clip-rule="evenodd"></path>
                                        </svg>
                                    </button>
                                    <div class="comment-menu-open-wrap hidden" aria-labelledby="main-wishlist-content-menu" role="menu" tabindex="0" data-headlessui-state="open">
                                        <div class="comment-menu-open-container" role="none">
                                            <div class="comment-menu-open-divison" role="none">
                                                <button class="comment-menu-open-choice ${reply.id} ${reply.wishlist_id}" type="button" id="comment-menu-open-update" role="menuitem" tabindex="-1">수정</button>
                                                <button class="comment-menu-open-choice ${reply.id} ${reply.wishlist_id}" type="button" id="comment-menu-open-delete" role="menuitem" tabindex="-1">삭제</button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                `
            }
            text += `
                            </div>
                        </div>
                        <!-- 댓글 구분선 -->
                        <div class="comment-line"></div>
                    </div>
                </div>
            `
        })
    return text;
}

// 댓글 보기를 눌렀을 때 위시리스트별 댓글 보여주기 실행
div.addEventListener("click", async (e)  => {
    // console.log(e.target.classList[0])
    // 위시리스트 댓글 열기 클릭 시 이벤트
    if(e.target.classList[0] === 'reply-open-button'){
        const wishlistId = e.target.classList[1]
        const commentOpen = document.getElementById(`open${wishlistId}`)
        const commentClose = document.getElementById(`close${wishlistId}`)
        const comment = document.getElementById(`reply-form${wishlistId}`)

        commentOpen.classList.add("hidden");
        commentClose.classList.remove("hidden");
        comment.classList.remove("hidden");

        // 댓글이 열렸을 때 댓글 리스트 보여주기
        await wishlistService.replygetList(wishlistId, replyshowList).then((replies) => {
            comment.innerHTML = replies;
            addClickEventReplyProfile()
            // console.log(replies)
        })

    // 위시리스트 게시글 댓글 닫기 버튼 클릭 시 이벤트
    } else if(e.target.classList[0] === 'reply-close-button') {
        const wishlistId = e.target.classList[1]
        const commentOpen = document.getElementById(`open${wishlistId}`)
        const commentClose = document.getElementById(`close${wishlistId}`)
        const comment = document.getElementById(`reply-form${wishlistId}`)

        commentOpen.classList.remove("hidden");
        commentClose.classList.add("hidden");
        comment.classList.add("hidden");
    }
})


// 위시리스트 게시글 작성 모달창 열기/닫기
const wishlistCreate = document.querySelector(".extra-create-button")
const modalCreateInput = document.querySelector(".post-create")
const modalCreateClose = document.querySelector(".create-close-container")
const modalCreateFinish = document.querySelector(".create-finish-container")

wishlistCreate.addEventListener("click", () => {
    modalCreateInput.classList.remove("hidden");
});
modalCreateClose.addEventListener("click", () => {
    modalCreateInput.classList.add("hidden");
});


// 위시리스트 작성 모달창에서 발행하기 버튼 클릭 시 작성 완료
modalCreateFinish.addEventListener("click", async () => {
    const wishlistContent = document.getElementById("wishlist-content")
    const isPrivate = document.getElementById("is-private")
    const wishlistCategory = document.getElementById("category");
    const tagElements = document.querySelectorAll('.create-tag-list span');
    const tagNames = Array.from(tagElements).map(span => span.textContent.trim());
    await wishlistService.write({
        is_private: isPrivate.value,
        wishlist_content: wishlistContent.value,
        category_id: wishlistCategory.value,
        tag_name: tagNames
    });

    // 모달 내용 초기화
    wishlistContent.value = "";
    isPrivate.value = "1";
    wishlistCategory.value = "100";
    // tagElements.innerHTML = "";
    // tagElements.forEach(element => {
    // element.innerHTML = "";
    // });
    tagElements.forEach(span => {
    const div = span.parentNode;
        if (div) {
            div.remove();
        }
    });

    // 작성된 게시글이 포함된 위시리스트 다시 화면에 보여주기
    const text = await wishlistService.getList(myWishlistIdCheck, page, category, keyword, showList);
    div.innerHTML = text;
    addClickEventWishlistProfile();
    addMoreButton();
    addAllevents();
    menuOpenevents();

    modalCreateInput.classList.add("hidden");
});



// 위시리스트 게시글 작성시 태그 추가 함수
const createTagInput = document.querySelector(".create-tags-input-container .create-tags-input");
const createTag = document.querySelector(".create-tags-wrap .create-tag");
const createTagError = document.querySelector(".create-tag-error");
const addTag = (e) => {
    const createTagDiv = document.createElement("div");
    createTagDiv.classList.add("create-tag-list");
    createTagDiv.innerHTML = `
        <span style="display: flex; align-items: center;">
            ${e.target.value}
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14px">
              <line x1="15" y1="5" x2="5" y2="15"></line>
              <line x1="5" y1="5" x2="15" y2="15"></line>
            </svg>
        </span>`;
    createTag.appendChild(createTagDiv);
    e.target.value = "";
    createTagError.classList.add("hidden")

    // 생성된 태그 div 클릭 시 태그 삭제 이벤트
    createTagDiv.addEventListener("click", (e) => {
        e.stopPropagation();
        createTag.removeChild(createTagDiv);
        createTagError.classList.add("hidden")
    });
}
createTagInput.addEventListener("keyup", (e) => {
    if (e.keyCode === 13) {
        const createTags = document.querySelectorAll(".create-tags-wrap .create-tag-list");

        // 입력 값의 길이가 10 이하이며, 목록의 태그 개수가 5이하면 태그 생성
        if (e.target.value.length <= 10 && createTags.length <= 5) {
            if (e.target.value) {
                addTag(e);
            }
        //에러 메시지 출력
        } else {
            createTagError.classList.remove("hidden")
        }
    }
});


// 여백 클릭 시 위시리스트 생성 모달창 닫기
const postCreateWrap = document.querySelector(".post-create-wrap");
document.addEventListener("click", (e) => {
    if (!wishlistCreate.contains(e.target) && !postCreateWrap.contains(e.target)) {
        modalCreateInput.classList.add("hidden");

        const wishlistContent = document.getElementById("wishlist-content")
        const isPrivate = document.getElementById("is-private")
        const category = document.getElementById("category")
        const tagElements = document.querySelector('.create-tag-list');
        // 모달창 닫았을 때 내용 초기화
        wishlistContent.value = "";
        isPrivate.value = "1";
        category.value = "100";
        tagElements.innerHTML = "";
    }
});

// 위시리스트 게시글 수정시 태그 추가 함수
const tagInput = document.querySelector(".update-tags-input-container .update-tags-input");
const tag = document.querySelector(".update-tags-wrap .tag");
const tagError = document.querySelector(".tag-error-text");

const modifyWishlistTag = (value) => {
    // console.log('호출됨', value)
    const tagDiv = document.createElement("div");
    tagDiv.classList.add("tag-list");
    tagDiv.innerHTML = `
        <span style="display: flex; align-items: center;">
            ${value}
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14px">
              <line x1="15" y1="5" x2="5" y2="15"></line>
              <line x1="5" y1="5" x2="15" y2="15"></line>
            </svg>
        </span>`;
    tag.appendChild(tagDiv);
    document.querySelector("input.update-tags-input").value = "";
    tagError.classList.add("hidden")

    // tagDiv 클릭 시 이벤트 발생
    tagDiv.addEventListener("click", (e) => {
        // 태그 삭제
        e.stopPropagation();
        tag.removeChild(tagDiv);
        tagError.classList.add("hidden")
    });
}


// 위시리스트 수정/삭제하기
div.addEventListener("click", async (e) => {
    // 수정 버튼 클릭시 수정 모달창 생성
    if(e.target.classList[0] === 'post-menu-open-choice'){
        console.log("수정")
        const wishlistId = e.target.classList[1]
        // console.log(wishlistId)

        /// 위시리스트 내용 담기
        const updateText = document.querySelector(`.update-category-input`)
        const pastText = document.querySelector(`.post-title-text${e.target.classList[1]}`)
        updateText.value = pastText.textContent
        // console.log(commentText.textContent)
        // console.log(commentText)

        // 카테고리 담기
        const pastCategory = document.querySelector(`.post-category${e.target.classList[1]}`).id
        const updateCategory = document.querySelector(`#update-category option[value="${pastCategory}"]`);
        // console.log(pastCategory)
        updateCategory.selected = true;

        // 공개/비공개 여부 담기
        const isPrivate = document.querySelector(`.member-name${e.target.classList[1]}`).id
        // console.log(isPrivate)
        let pastPrivate = isPrivate === 'true' ? 1 : 0
        // console.log(pastPrivate)
        const updatePrivate = document.querySelector(`#update-private option[value="${pastPrivate}"]`);
        updatePrivate.selected = true;

        // 이전에 있던 태그들 각각 반복으로 태그박스 만들어서 담기
        await wishlistService.getList(myWishlistIdCheck, page, category, keyword).then((data) => {
            const existingTags = data.tags[wishlistId]
            // console.log(existingTags)
            existingTags.forEach((tag) => {modifyWishlistTag(tag);});
        })

        // 위시리스트 수정 모달 발행하기 버튼 클릭시 수정 완료
        const modalPostUpdateFinish = document.querySelector(".update-finish-botton");
        modalPostUpdateFinish.addEventListener("click", async () => {
            wishListId = e.target.classList[1]
            const updatedContent = document.getElementById("update-content").value;
            const updatedCategory = document.getElementById("update-category").value;
            const updatedPrivate = document.getElementById("update-private").value;
            const tagElements = document.querySelectorAll('.tag-list span');
            const updateTagNames = Array.from(tagElements).map(span => span.textContent.trim());
            // console.log(tagElements)
            // console.log(updateTagNames)

            // 수정한 값을 가져와서 newWishlist에 할당
            const newWishlist = {
                wishlist_id: wishListId,
                wishlist_content: updatedContent,
                category_id: updatedCategory,
                is_private: updatedPrivate,
                tag_name: updateTagNames,
            };

            // Update 함수를 호출하여 수정된 위시리스트 정보 전송
            await wishlistService.wishlistUpdate(wishListId, newWishlist);

            // 수정된 게시글을 포함한 위시리스트 게시글 다시 뿌려주기
            const text = await wishlistService.getList(myWishlistIdCheck, page, category, keyword, showList);
            div.innerHTML = text;

            addMoreButton()
            addClickEventWishlistProfile()
            addAllevents();
            menuOpenevents();
        });

    // 삭제 버튼 클릭시 삭제하기 (status =0)
    } else if (e.target.id === 'post-menu-open-delete'){
        console.log("삭제")
        const wishlistId = e.target.classList[1]
        // console.log(wishlistId)
        await wishlistService.wishlistRemove(wishlistId);
        page = 1
        const text = await wishlistService.getList(myWishlistIdCheck, page, category,keyword, showList);
        div.innerHTML = text;
        addClickEventWishlistProfile()
        addAllevents();
        menuOpenevents();
        addMoreButton();
    }
})

//댓글 등록 버튼 클릭 시 작성 완료
// div.addEventListener("click", async (e) => {
//     if (e.target.id === 'comment-upload-button') {
//         // console.log(e.target)
//         const replyBtn = document.getElementById("comment-upload-button")
//         const wishlistId = document.querySelector(".reply-open-button").classList[1]
//         replyBtn.addEventListener("click", async () => {
//             const replyContent = document.getElementById("reply-content")
//             // const wishlistId = replyBtn.classList[1]
//             const comment = document.getElementById(`reply-form${wishlistId}`)
//
//             // Write 함수를 호출하여 작성된 댓글 정보 전송
//             await wishlistService.replyWrite({
//                 reply_content: replyContent.value,
//                 wishlist_id: wishlistId
//             });
//
//             // 작성칸 내용 초기화
//             replyContent.value = "";
//
//             // 수정된 댓글을 포함한 댓글 리스트 다시 뿌려주기
//             await wishlistService.replygetList(wishlistId, replyshowList).then((replies) => {
//                 comment.innerHTML = replies;
//             }).then(async () => {
//                 await addClickEventReplyProfile();
//                 await addMoreButton();
//             })
//         })
//     }
// });

//댓글 등록 버튼 클릭 시 작성 완료
div.addEventListener("click", async (e) => {
    if (e.target.id === 'comment-upload-button') {
        const wishlistPost = e.target.closest(".wishlist-post")
        const replyBtn = wishlistPost.querySelector("#comment-upload-button")
        const wishlistId = wishlistPost.querySelector(".comment-all-wrap").classList[1]
        console.log(wishlistId)
        // console.log(wishlistPost.querySelector(".comment-all-wrap"))
        // const replyBtn = document.getElementById("comment-upload-button")
        // const wishlistId = document.querySelector(".reply-open-button").classList[1]
        replyBtn.addEventListener("click", async () => {
            const replyContent = document.getElementById("reply-content")
            // const wishlistId = replyBtn.classList[1]
            const comment = document.getElementById(`reply-form${wishlistId}`)
            await wishlistService.replyWrite({
                reply_content: replyContent.value,
                wishlist_id: wishlistId
            });

            console.log(replyContent)

            replyContent.value = "";

            await wishlistService.replygetList(wishlistId, replyshowList).then((replies) => {
                comment.innerHTML = replies;
            }).then(async () => {
                await addClickEventReplyProfile();
                await addMoreButton();
            })
        })
    }
});

// 댓글 메뉴 버튼 생성하기
div.addEventListener("click", async (e) => {
    // 수정하기
    if (e.target.classList[0] === 'comment-menu-icon') {
        // console.log(e.target)
        // 위시리스트 댓글 메뉴 열고 닫기
        const wishlistCommentMenuButton = document.querySelector(".comment-menu");
        const wishlistCommentMenu = document.querySelector(".comment-menu-open-wrap");

        // 위시리스트 댓글 버튼 클릭 시 이벤트
        wishlistCommentMenuButton.addEventListener("click", () => {
            wishlistCommentMenu.classList.toggle("hidden");
        });

        // 여백 클릭 시 댓글 메뉴 닫기
        document.addEventListener("click", (e) => {
            if (!wishlistCommentMenuButton.contains(e.target) && !wishlistCommentMenu.contains(e.target)) {
                wishlistCommentMenu.classList.add("hidden");
            }
        });
    }
})

// 댓글 메뉴 버튼을 눌렀을 때 수정/삭제
div.addEventListener("click", async (e) => {
    // 수정 버튼 클릭시 수정하기
    if (e.target.id === 'comment-menu-open-update') {
        wishlistId = e.target.classList[2]
        const commentMenuOpenUpdate = document.getElementById("comment-menu-open-update");
        const commentInputUpdate = document.querySelector(".comment-update-box-all-wrap");
        const commentComment = document.querySelector(".comment-list-all-wrap");
        const commentUploadFinish =document.getElementById("comment-update-upload");
        const commenmtUpdateArea =document.getElementById("comment-input-update-guide");
        const replyUpdateBtn = document.querySelector(".comment-update-upload-container")
        const comment = document.getElementById(`reply-form${wishlistId}`)
        // 입력한 정보 확인용
        commenmtUpdateArea.addEventListener("keyup", () => {
            console.log(commenmtUpdateArea.value)
        })

        // 댓글 수정에서 등록 버튼 눌렀을 때 등록 완료
        let commentText = document.querySelector(`.comment-text${e.target.classList[1]}`)
        commenmtUpdateArea.value = commentText.innerText
        replyUpdateBtn.addEventListener("click", async () => {
            // console.log(e.target.classList[1])
            newReply = commenmtUpdateArea.value
            replyId = e.target.classList[1]

            // Update 함수를 호출하여 수정된 댓글 정보 전송
            wishlistService.replyUpdate(replyId, newReply)

            // 수정된 댓글 정보를 포함한 댓글 리스트 뿌려주기
            await wishlistService.replygetList(wishlistId, replyshowList).then((replies) => {
                comment.innerHTML = replies;
            }).then(async () => {
                await addClickEventReplyProfile();
                await addMoreButton();
            })
        })

        // 댓글 수정 입력 활성화 후 개별 댓글 숨기기
        commentMenuOpenUpdate.addEventListener("click", () => {
            commentInputUpdate.classList.remove("hidden");
            commentComment.classList.add("hidden");

        });
        // 댓글 수정 입력 비활성화 개별 댓글 활성화
        commentUploadFinish.addEventListener("click", () => {
            commentInputUpdate.classList.add("hidden");
            commentComment.classList.remove("hidden");
        });

    // 삭제 버튼 클릭시 삭제하기 (status =0)
    } else if (e.target.id === 'comment-menu-open-delete') {
        const  replyId = e.target.classList[1]
        const wishlistId = e.target.classList[2]
        const comment = document.getElementById(`reply-form${wishlistId}`)
        // console.log(replyId)
        // console.log(wishlistId)

        // Remove 함수를 호출하여 전달된 정보 삭제(status=0 으로 update)
        await wishlistService.replyRemove(replyId)

        // 삭제된 댓글을 제외한 댓글 리스트 다시 뿌려주기
        await wishlistService.replygetList(wishlistId, replyshowList).then((replies) => {
            comment.innerHTML = replies;
        }).then(async () => {
            await addClickEventReplyProfile();
            await addMoreButton();
        })
    }
})



// 좋아요 아이콘 클릭 시 화면 반영하기
div.addEventListener('click', async (e)=> {
    const button = e.target.closest('.post-like-container')

    if(button){
        if(memberId == 0){
            window.location.href = '/member/login/'
            return;
        }
        let emptyHeartIcon = button.querySelector('.post-like-icon')
        let fullHeartIcon = button.querySelector(".post-like-icon.liked")
        let displayStyle = window.getComputedStyle(emptyHeartIcon).display
        emptyHeartIcon.style.display = displayStyle === 'none' ? 'block':'none';
        fullHeartIcon.style.display = displayStyle === 'none'? 'none': 'block';

        let wishlistId = button.classList[1]
        // console.log(wishlistId)

        const postLike = await wishlistService.likeWishlist(wishlistId, memberId, displayStyle)
        const likeCountContainer = button.closest('.post-items-wrap').querySelector('.post-like-count')
        if(likeCountContainer){
            likeCountContainer.innerText = postLike.totalLikeCount
        }
    }
})


//프로필 사진 눌렀을 때 틴친, 쪽지 모달 시작
const sendLetterBoxBtn = document.querySelector(".send-letter-btn");
const sendLetter = document.querySelector(".send-modal-wrap");
const senderInfo = document.querySelector(".send-sender-email")
const receiverInfo = document.querySelector(".send-receiver-email")
const profile = document.querySelector(".profile");


// 쪽지 모달창에 발신자 정보 받아와서 넣기
const sendLetterAddInfo = (wishlistId) => {
    const memberName = document.querySelector("input[name=member-name]").value;
    const memberEmail = document.querySelector("input[name=member-email]").value;
    senderInfo.innerText = `${memberName} (${memberEmail})`;
    const receiverName = document.querySelector(`.member-name${wishlistId}`).innerText;
    const receiverEmail = document.querySelector(`.member-email${wishlistId}`).value;
    receiverInfo.innerText = `${receiverName} (${receiverEmail})`;
}

// 틴친 신청 버튼
const teenchinAddButton = document.querySelector(".teenchin-add-btn");
// 틴친 신청취소 버튼
const teenchinCancelButton = document.querySelector(".teenchin-request-btn");
// 틴친 수락/거절 버튼
const teenchinAcceptButton = document.querySelector(".teenchin-accept-btn");
// 틴친 끊기 버튼
const teenchinDeleteButton = document.querySelector(".teenchin-btn");

// // 틴친 신청 버튼
// const teenchinAddButton = document.querySelector(".teenchin-add-btn");
// // 틴친 신청취소 버튼
// const teenchinCancelButton = document.querySelector(".teenchin-request-btn");
// // 틴친 끊기 버튼
// const teenchinDeleteButton = document.querySelector(".teenchin-btn");

// 보여주는 함수
const helpShowButton = (button) => {
    if (button.classList.contains("hidden")) {
        button.classList.remove("hidden");
    }
}
// 숨겨주는 함수
const helpHideButton = (button) => {
    if (!button.classList.contains("hidden")) {
        button.classList.add("hidden");
    }
}


// 이제 위 요소들을 사용하여 틴친 상태에 따라 버튼을 바꿔줄 함수 정의
const showButtonsByTeenchinStatus = (teenchinStatus) => {
    let status = teenchinStatus.teenchinStatus;
    let isSender = teenchinStatus.isSender;
    if (status === 0) {
        helpShowButton(teenchinAddButton);
        helpHideButton(teenchinCancelButton);
        helpHideButton(teenchinAcceptButton)
        helpHideButton(teenchinDeleteButton);
    } else if (status === 1) {
        helpShowButton(teenchinDeleteButton);
        helpHideButton(teenchinAddButton);
        helpHideButton(teenchinAcceptButton)
        helpHideButton(teenchinCancelButton);
    } else if (isSender) {
        helpShowButton(teenchinCancelButton);
        helpHideButton(teenchinAcceptButton);
        helpHideButton(teenchinAddButton);
        helpHideButton(teenchinDeleteButton);
    } else {
        helpShowButton(teenchinAcceptButton);
        helpHideButton(teenchinAddButton);
        helpHideButton(teenchinCancelButton);
        helpHideButton(teenchinDeleteButton);
    }
}


// // 틴친 클릭 시 프로필 모달 나오기
// const profileModal = document.querySelector("div.profile");
// const profileModalProfileImage = document.querySelector(".profile-default-img");
// const profileModalMemberName = document.querySelector("div.profile-name");
// let opponentTeenchinId = 0;
// const showMemberProfileModal = async (wishlistId) => {
//     if (!loginCheck) return;
//     opponentTeenchinId = document.querySelector(`.wishlist-writer-id${wishlistId}`).value;
//     // console.log(opponentTeenchinId)
//     // console.log(loginId)
//     if (profileModal.classList.contains("hidden") && (opponentTeenchinId !== memberId)) {
//         profileModal.classList.remove("hidden")
//         const memberProfileImage = document.querySelector(`.profile-image${wishlistId}`);
//         profileModalProfileImage.src = memberProfileImage.src;
//         const memberProfileName = document.querySelector(`.member-name${wishlistId}`);
//         profileModalMemberName.innerText = memberProfileName.innerText;
//         sendLetterAddInfo(wishlistId);
//
//         await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
//     }
// }

// 위에서 정의한 함수를 사용할 때, 댓글에서 프로필 사진을 클릭하면
// 해당 멤버의 id를 같이 넘겨 틴친 상태에 따라 버튼을 바로 바꿔줘야 합니다.
// 따라서 프로필 모달이 표시됨과 동시에 이루어지도록 합니다.

// 틴친 클릭 시 프로필 모달 나오도록 하기
const profileModal = document.querySelector("div.profile");
const profileModalProfileImage = document.querySelector(".profile-default-img");
const profileModalMemberName = document.querySelector("div.profile-name");
let opponentTeenchinId = 0;
const showMemberProfileModal = async (wishlistId) => {
    opponentTeenchinId = document.querySelector(`.wishlist-writer-id${wishlistId}`).value;
    if (memberId === opponentTeenchinId) return;
    if (profileModal.classList.contains("hidden")) {
        profileModal.classList.remove("hidden")
        const memberProfileImage = document.querySelector(`.profile-image${wishlistId}`);
        profileModalProfileImage.src = memberProfileImage.src;
        const memberProfileName = document.querySelector(`.member-name${wishlistId}`);
        profileModalMemberName.innerText = memberProfileName.innerText;
        sendLetterAddInfo(wishlistId);
        await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
    }
}



// 여백 클릭시 프로필 모달 닫기
const hideMemberProfileModal = () => {
    if (!profileModal.classList.contains("hidden")){
        profileModal.classList.add("hidden");
    }
}


// 댓글의 프로필 클릭시 프로필 모달 보여주기
const addClickEventReplyProfile = () => {
    const profilePhotos = document.querySelectorAll(".comment-profile-container");
    profilePhotos.forEach((wrap) => {
        wrap.addEventListener("click", (e) => {
            let wishlistId = wrap.classList[1];
            showMemberProfileModal(wishlistId);
        })
    })
    const modalDivision = document.querySelector(".modal-divison")
    const modalContainer = document.querySelector(".teenchin-box.post-update-wrap")
    modalDivision.addEventListener("click", (e) => {
        if (e.target !== modalContainer){
            hideMemberProfileModal()
        }
    })
}


//쪽지 보내기 버튼 클릭시 쪽지 보내기 모달 출력
sendLetterBoxBtn.addEventListener("click", () => {
    profile.classList.add("hidden");
    sendLetter.classList.remove("hidden");
})


// 쪽지 보내기 닫기(버튼) 모달 이벤트
const sendLetterCloseBtn = document.querySelector(".send-close-btn");

if (sendLetterCloseBtn){
    sendLetterCloseBtn.addEventListener("click", () => {
        sendLetter.classList.add("hidden");
    });
}


// 쪽지 보내기 닫기(여백) 모달 이벤트
const sendLetterModal = document.querySelector(".send-modal-box");

if (sendLetterModal){
    document.addEventListener("click", (e) => {
        if (!sendLetterBoxBtn.contains(e.target) && !sendLetterModal.contains(e.target)) {
            sendLetter.classList.add("hidden");
        }
    });
}


// 쪽지 보내기 모달 이벤트
const sendLetterBtn = document.querySelector(".send-check-btn");

if (sendLetterBtn){
    sendLetterBtn.addEventListener("click", async () => {
        const letterContent = document.getElementById("letter-content").value;
        if (!letterContent) return;
        const receiverId = document.querySelector(".send-receiver-email").innerText;
        await activityLetterService.write({
            letter_content: letterContent,
            receiver_id: receiverId
        })
        Swal.fire("쪽지가 전송 되었습니다.", "", "success");
    });
}


// 틴친 추가 모달 이벤트
const teenFriendAdd = document.querySelector(".teenchin-add-btn");
const teenFriendRequest = document.querySelector(".teenchin-request-btn");

if (teenFriendAdd){
    teenFriendAdd.addEventListener("click", () => {
        Swal.fire({
            title: "틴친 신청을 보낼까요?",
            text: "",
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#CE201B",
            cancelButtonColor: "#E1E1E1",
            confirmButtonText: "친구추가",
            cancelButtonText: "닫기",
        }).then(async (result) => {
            if (result.value) {
                await activityTeenchinService.apply(opponentTeenchinId);
                await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
                Swal.fire("틴친 신청을 보냈어요!", "", "success");
            } else if ((result.dismiss = "cancel")) {
                return;
            }
        });
    });
}

// 틴친 신청 취소 모달 이벤트
if (teenFriendRequest){
    teenFriendRequest.addEventListener("click", () => {
        Swal.fire({
            title: "신청을 취소할까요?",
            text: "",
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#CE201B",
            cancelButtonColor: "#E1E1E1",
            confirmButtonText: "신청취소",
            cancelButtonText: "닫기",
        }).then(async (result) => {
            if (result.value) {
                await activityTeenchinService.cancelOrAcceptDenyTeenchin(opponentTeenchinId, true, false);
                await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
                Swal.fire("틴친 신청이 취소되었습니다.", "", "success");
            } else if ((result.dismiss = "cancel")) {
                return;
            }
        });
    });
}

// 틴친 수락/거절 모달 이벤트
if (teenchinAcceptButton) {
    teenchinAcceptButton.addEventListener("click", () => {
        Swal.fire({
            title: "신청을 수락할까요?",
            text: "",
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#CE201B",
            cancelButtonColor: "#E1E1E1",
            confirmButtonText: "수락",
            cancelButtonText: "거절",
        }).then(async (result) => {
            if (result.value) {
                await activityTeenchinService.cancelOrAcceptDenyTeenchin(opponentTeenchinId, false, true);
                await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
                Swal.fire("틴친신청을 수락했습니다.", "", "success");
            } else if ((result.dismiss = "cancel")) {
                await activityTeenchinService.cancelOrAcceptDenyTeenchin(opponentTeenchinId, false, false);
                await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
                Swal.fire("틴친신청을 거절했습니다.", "", "success");
                return;
            }
        });
    })
}

// 틴친 취소 모달 이벤트
const teenFriendCancle = document.querySelector(".teenchin-btn");

if (teenFriendCancle){
    teenFriendCancle.addEventListener("click", () => {
        Swal.fire({
            title: "틴친을 그만둘까요?",
            text: "",
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#CE201B",
            cancelButtonColor: "#E1E1E1",
            confirmButtonText: "틴친끊기",
            cancelButtonText: "닫기",
        }).then(async (result) => {
            if (result.value) {
                await activityTeenchinService.removeTeenchin(opponentTeenchinId);
                await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
                Swal.fire("틴친 관계가 해제되었어요.", "", "success");
            } else if ((result.dismiss = "cancel")) {
                return;
            }
        });
    });
}


// // 틴친 추가 모달 이벤트
// const teenFriendAdd = document.querySelector(".teenchin-add-btn");
// const teenFriendRequest = document.querySelector(".teenchin-request-btn");
//
// if (teenFriendAdd){
//     teenFriendAdd.addEventListener("click", () => {
//         Swal.fire({
//             title: "틴친 신청을 보낼까요?",
//             text: "",
//             icon: "warning",
//             showCancelButton: true,
//             confirmButtonColor: "#CE201B",
//             cancelButtonColor: "#E1E1E1",
//             confirmButtonText: "친구추가",
//             cancelButtonText: "닫기",
//         }).then(async (result) => {
//             if (result.value) {
//                 await activityTeenchinService.apply(opponentTeenchinId);
//                 await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
//             } else if ((result.dismiss = "cancel")) {
//                 return;
//             }
//         });
//     });
// }
//
//
// // 틴친 신청 취소 모달 이벤트
// if (teenFriendRequest){
//     teenFriendRequest.addEventListener("click", () => {
//         Swal.fire({
//             title: "신청을 취소할까요?",
//             text: "",
//             icon: "warning",
//             showCancelButton: true,
//             confirmButtonColor: "#CE201B",
//             cancelButtonColor: "#E1E1E1",
//             confirmButtonText: "신청취소",
//             cancelButtonText: "닫기",
//         }).then(async (result) => {
//             if (result.value) {
//                 await activityTeenchinService.cancelApplyTeenchin(opponentTeenchinId);
//                 await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
//             } else if ((result.dismiss = "cancel")) {
//                 return;
//             }
//         });
//     });
// }
//
// // 틴친 취소 모달 이벤트
// const teenFriendCancle = document.querySelector(".teenchin-btn");
//
// if (teenFriendCancle){
//     teenFriendCancle.addEventListener("click", () => {
//         Swal.fire({
//             title: "틴친을 그만둘까요?",
//             text: "",
//             icon: "warning",
//             showCancelButton: true,
//             confirmButtonColor: "#CE201B",
//             cancelButtonColor: "#E1E1E1",
//             confirmButtonText: "틴친끊기",
//             cancelButtonText: "닫기",
//         }).then(async (result) => {
//             if (result.value) {
//                 await activityTeenchinService.removeTeenchin(opponentTeenchinId);
//                 await activityTeenchinService.getTeenchinStatus(opponentTeenchinId, showButtonsByTeenchinStatus);
//             } else if ((result.dismiss = "cancel")) {
//                 return;
//             }
//         });
//     });
// }


// 마이페이지에서 넘어오면서 클릭한 위시리스트가 제일 위에 보일때 테두리 강조하기
const findFirst = () => {
        const trueId = document.querySelector('.post-top-warp')
        const flaseId = document.querySelector('.wishlist-post')
        // console.log(trueId)
        // console.log(flaseId)
        if(trueId.classList[1] === flaseId.classList[1]) {
            const addPost = document.querySelector('.wishlist-post:first-child');
            // console.log(addPost)
            if (addPost) {
                addPost.style.borderWidth = '5px';
            }
        }
    }


//작성 날짜 계산하기
function timeForToday(datetime) {
    const today = new Date();
    const date = new Date(datetime);

    let gap = Math.floor((today.getTime() - date.getTime()) / 1000 / 60);

    if (gap < 1) {
        return "방금 전";
    }

    if (gap < 60) {
        return `${gap}분 전`;
    }

    gap = Math.floor(gap / 60);

    if (gap < 24) {
        return `${gap}시간 전`;
    }

    gap = Math.floor(gap / 24);

    if (gap < 31) {
        return `${gap}일 전`;
    }

    gap = Math.floor(gap / 31);

    if (gap < 12) {
        return `${gap}개월 전`;
    }

    gap = Math.floor(gap / 12);

    return `${gap}년 전`;
}