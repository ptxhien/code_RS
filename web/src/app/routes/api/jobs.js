const express = require("express");
const router = express.Router();
const JobController = require("../../controllers/api/job");

router.get("/", JobController.get);


module.exports = router;
