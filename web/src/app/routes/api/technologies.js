const express = require("express");
const router = express.Router();
const TechnologyController = require("../../controllers/api/technology");

router.get("/", TechnologyController.get);


module.exports = router;
