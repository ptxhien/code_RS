const express = require("express");
const router = express.Router();

router.get("/", (req, res) => {
  res.render("sites/events/index", {
    title: "Events",
    css: "dynamic_css/events",
    js: "dynamic_js/events",
    nav: "events",
    breadcrumb: [
      { name: "Home", link: "/" },
      { name: "Events", link: "" },
    ],
  });
});

router.get("/:id", (req, res) => {
  res.render("sites/events/detail", {
    title: "Sample event",
    css: "dynamic_css/event",
    js: "dynamic_js/event",
    nav: "events",
    breadcrumb: [
      { name: "Home", link: "/" },
      { name: "Events", link: "/events" },
      { name: "Sample event", link: "" },
    ],
  });
});

module.exports = router;
