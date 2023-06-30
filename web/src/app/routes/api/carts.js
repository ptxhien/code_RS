const express = require("express");
const router = express.Router();
const CartController = require("../../controllers/api/cart");
const { apiVerify } = require("../../../utilities/middlewares/verify");


router.get("/", apiVerify, CartController.get);
router.post("/", apiVerify, CartController.post);

module.exports = router;
