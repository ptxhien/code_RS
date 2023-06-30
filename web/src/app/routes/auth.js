const express = require("express");
const router = express.Router();
const authController = require("../controllers/auth")
const {logout, redirectToUser} = require("../../utilities/middlewares/verify");


router.get("/register", redirectToUser, authController.registry);

router.post("/register", redirectToUser, authController.submit);

router.post("/login", redirectToUser, authController.login);

router.post("/logout", logout, authController.logout);

// router.get("/resetall", authController.hashAll);


module.exports = router;