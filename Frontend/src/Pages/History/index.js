import React, { Fragment } from "react";
import {
  Row,
  Col,
  Button,
  CardHeader,
  Container,
  Card,
  CardBody,
  Progress,
  ListGroup,
  ListGroupItem,
  CardFooter,
  CustomInput,
  Input,
  Dropdown,
  DropdownItem,
  DropdownToggle,
  DropdownMenu,
  UncontrolledButtonDropdown,
} from "reactstrap";
import { useDispatch, useSelector } from "react-redux";

import ThemeOptions from "../../Layout/ThemeOptions/";
import AppHeader from "../../Layout/AppHeader/";

import "../History/style.css";

export default function History() {
  const dispatch = useDispatch();

  return (
    <Fragment>
      <ThemeOptions />
      <AppHeader />
    </Fragment>
  );
}
