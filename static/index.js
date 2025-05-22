const navItems = document.querySelectorAll(".nav-item-custom");

navItems.forEach((item) => {
  item.addEventListener("click", () => {
    navItems.forEach((i) => i.classList.remove("nav-item-active"));
    item.classList.add("nav-item-active");
  });
});
