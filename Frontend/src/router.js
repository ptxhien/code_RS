import React, { Suspense, lazy, Fragment } from "react";
import LoginPage from "./Pages/Login";
import RegisterPage from "./Pages/Register";
import HomePage from "./Pages/Home";
import Dashboard from "./Pages/Dashboards";
import History from "./Pages/History";
import CourseDetail from "./Pages/CourseDetail";
import EditProfile from "./Pages/Accounts/EditProfile";
import Cart from "./Pages/Cart";

const routes = [
  {
    path: "/",
    exact: true,
    main: () => <HomePage></HomePage>,
    auth: false,
  },
  {
    path: "/course/:id",
    main: () => <CourseDetail />,
    auth: false,
  },
  {
    path: "/login",
    exact: true,
    main: () => <LoginPage></LoginPage>,
    auth: false,
  },
  {
    path: "/register",
    exact: true,
    main: () => <RegisterPage></RegisterPage>,
    auth: false,
  },
  {
    path: "/dashboard",
    exact: true,
    main: () => <History></History>,
    auth: false,
  },
  {
    path: "/update-profile",
    exact: true,
    main: () => <EditProfile></EditProfile>,
    auth: false,
  },
  {
    path: "/cart",
    exact: true,
    main: () => <Cart />,
    auth: false,
  },
];

export default routes;
