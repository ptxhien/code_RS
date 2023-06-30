const express = require("express");
const { redirect } = require("../../utilities/middlewares/verify");
const router = express.Router();


router.get("/", redirect, (req, res, next) => {
  res.status(200).redirect("/profile");
  return;
});

router.get("/profile", redirect, (req, res, next) => {
  res.render("sites/user/profile", {
    title: "Register",
    css: "dynamic_css/register",
    js: "dynamic_js/register",
    nav: "profile",
    layout: "register",
    breadcrumb: [],
  });
});

router.get("/history",);

module.exports = router;
