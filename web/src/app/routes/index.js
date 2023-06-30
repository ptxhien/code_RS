const express = require("express");
const courses = require("./courses");
const blogs = require("./blogs");
const events = require("./events");
const auth = require("./auth");
const user = require("./user");
const api = require("./api");
const { redirect, redirectToUser } = require("../../utilities/middlewares/verify");

const route = (app) => {

  app.get("/about", (req, res) => {
    res.render("sites/about/index", {
      title: "About us",
      css: "dynamic_css/about",
      js: "dynamic_js/about",
      nav: "about",
      breadcrumb: [
        { name: "Home", link: "/" },
        { name: "About us", link: "" },
      ],
    });
  });

  app.get("/contact", (req, res) => {
    res.render("sites/contact/index", {
      title: "Contact",
      css: "dynamic_css/contact",
      js: "dynamic_js/contact",
      nav: "contact",
      breadcrumb: [
        { name: "Home", link: "/" },
        { name: "Contact", link: "" },
      ],
    });
  });

  app.use("/courses", courses);
  app.use("/blogs", blogs);
  app.use("/events", events);
  app.use("/auth", auth);
  app.use("/user", user);
  app.use("/api", api);

  app.get("/", (req, res) => {
    res.render("sites/home/index", {
      title: "Home",
      css: "dynamic_css/index",
      js: "dynamic_js/index",
      nav: "home",
    });
  });

  app.get("/unauthorization", redirectToUser, (req, res) => {
    res.render("sites/err/401", {
      title: "Unauthorization ",
      css: "dynamic_css/index",
      js: "dynamic_js/index",
      nav: "Unauthorization",
    });
  });

  app.get("*", (req, res) => {
    res.status(404).render("sites/err/404", {
      title: "Not found",
      css: "dynamic_css/index",
      js: "dynamic_js/index",
      nav: "Not found",
    });
  });
};

module.exports = { route };
