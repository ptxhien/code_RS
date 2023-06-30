const express = require("express");
const router = express.Router();
const RatingController = require("../../controllers/api/rating");
const { apiVerify } = require("../../../utilities/middlewares/verify");

router.post("/", apiVerify, RatingController.post);

module.exports = router;
