let page = 1
let keyword = ``
let category = 0
let ongoing = false
let includingEnd = false


//
const mainFestivalSearchInput = document.querySelector(".main-festival-search-input")

mainFestivalSearchInput.addEventListener("input", (e) => {
    keyword = e.target.value
    festivalService.getList(page, keyword, category, date, ongoing, includingEnd, createList.showList)
})

//
const mainFestivalCategorySelect = document.querySelector(".main-festival-category-select")

mainFestivalCategorySelect.addEventListener("change", (e) => {
    category = e.target.value
    festivalService.getList(page, keyword, category, date, ongoing, includingEnd, createList.showList)
})

const checkLeft = document.querySelector("#check-left")
checkLeft.addEventListener("change", (e) => {
    ongoing = !!e.target.checked;
    festivalService.getList(page, keyword, category, date, ongoing, includingEnd, createList.showList)
})

const checkRight = document.querySelector("#check-right")
checkRight.addEventListener("change", (e) => {
    includingEnd = !!e.target.checked;
    festivalService.getList(page, keyword, category, date, ongoing, includingEnd, createList.showList)
})

const createList = (() => {
   const showList = async () => {
       let text = ``;
       await pagination()
   }

   const pagination = () => {

   }

   return {showList: showList}
})()

festivalService.getList(page, keyword, category, date, ongoing, includingEnd, createList.showList)
