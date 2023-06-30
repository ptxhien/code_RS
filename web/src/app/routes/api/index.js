const express = require("express");
const router = express.Router();
const courses = require("./courses");
const auth = require("./auth");
const jobs = require("./jobs");
const languages = require("./languages");
const technologies = require("./technologies");
const freetime = require("./freetime");
const invoices = require("./invoices");
const rating = require("./rating");
const major = require("./major");
const carts = require("./carts");


router.use("/courses", courses);
router.use("/auth", auth);
router.use("/jobs", jobs);
router.use("/languages", languages);
router.use("/technologies", technologies);
router.use("/freetime", freetime);
router.use("/invoices", invoices);
router.use("/rating", rating);
router.use("/major", major);
router.use("/carts", carts);

module.exports = router;
