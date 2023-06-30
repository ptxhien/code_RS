const express = require("express");
const router = express.Router();
const authController = require("../../controllers/api/auth")


router.post("/register", authController.register);

router.post("/login", authController.login);

router.post("/update", authController.update);

module.exports = router;