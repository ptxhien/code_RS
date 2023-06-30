const freetimeController = require("../../controllers/api/freetime");
const express = require("express");
const router = express.Router();

router.get("/", freetimeController.get);

module.exports = router;
