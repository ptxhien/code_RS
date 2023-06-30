const express = require("express");
const router = express.Router();
const LanguageController = require("../../controllers/api/language");

router.get("/", LanguageController.get);


module.exports = router;
