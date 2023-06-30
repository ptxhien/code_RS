"use strict";

const jwt = require("jsonwebtoken");

const apiVerify = async function (req, res, next) {
  const token = req.headers.auth;

  if (!token) {
    res.locals.isLogged = false;
    res.locals.username = null;
    req.user = null;
    res.status(401).json({msg: "Unauthorized"});
    return;
  }

  try {
    const user = await jwt.verify(
      token,
      "zFUVn{;Sd4!]#lN" || "DevSecretKey"
    );
    req.user = user;
    res.locals.user = user;
    res.locals.isLogged = true;
    next();
  } catch (err) {
    res.locals.isLogged = false;
    res.locals.username = null;
    req.user = null;
    res.status(403).json({ msg: "Token is out dated, please login again"});
  }
};

const verify = async function (req, res, next) {
  const token = req.cookies.authToken;
  if (!token) {
    res.locals.isLogged = false;
    res.locals.username = null;
    req.user = null;
    next();
    return;
  }

  try {
    const user = await jwt.verify(
      token,
      "zFUVn{;Sd4!]#lN" || "DevSecretKey"
    );
    req.user = user;
    res.locals.user = user;
    res.locals.isLogged = true;
    next();
  } catch (err) {
    res.locals.isLogged = false;
    res.locals.username = null;
    req.user = null;
    next();
  }
};

const logout = function (req, res, next) {
  res.clearCookie("authToken");
  next();
};

const redirect = function (req, res, next) { 
  if (req.user){
    next();
  } else {
    res.status(401).redirect("/unauthorization");
    return;
  }
}

const redirectToUser = function (req, res, next) {
  if (req.user){
    res.status(200).redirect("/user/profile");
    return;
  } else{
    next();
  }
}

module.exports = { apiVerify, verify, logout, redirect, redirectToUser };
