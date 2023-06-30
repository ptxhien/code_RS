const express = require("express");
const router = express.Router();
const MajorController = require("../../controllers/api/major");

router.get("/", MajorController.get);


module.exports = router;
