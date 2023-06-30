const express = require("express");
const router = express.Router();
const APICousesController = require("../../controllers/api/courses");

router.get("/", APICousesController.get);
router.get("/:id", APICousesController.getOne);


module.exports = router;
