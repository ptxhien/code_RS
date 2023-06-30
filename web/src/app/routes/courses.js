const express = require("express");
const CourseController = require("../controllers/course");
const router = express.Router();

router.get("/", CourseController.get);
router.post("/", CourseController.recommend);

router.use("/:id", CourseController.getOne);


module.exports = router;
