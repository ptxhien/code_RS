const express = require("express");
const router = express.Router();

router.get("/", (req, res) => {
  res.render("sites/blogs/index", {
    title: "Blogs",
    css: "dynamic_css/blogs",
    js: "dynamic_js/blogs",
    nav: "blogs",
    breadcrumb: [
      { name: "Home", link: "/" },
      { name: "Blogs", link: "" },
    ],
  });
});

router.get("/:id", (req, res) => {
  res.render("sites/blogs/detail", {
    title: "Sample blog",
    css: "dynamic_css/blog",
    js: "dynamic_js/blog",
    nav: "blogs",
    breadcrumb: [
      { name: "Home", link: "/" },
      { name: "Blogs", link: "/blogs" },
      { name: "Sample blog", link: "" },
    ],
  });
});

module.exports = router;
