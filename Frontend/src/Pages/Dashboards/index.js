import React, { Fragment, useEffect, useLayoutEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import http from "../../redux/utils/http";
import * as Types from "./../../redux/constants/actionType";

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
  FormGroup,
  Label,
  UncontrolledButtonDropdown,
} from "reactstrap";
import ThemeOptions from "../../Layout/ThemeOptions/";
import AppHeader from "../../Layout/AppHeader/";

import "./style.scss";
import { Redirect } from "react-router";
import { getInvoices } from "../../redux/actions/invoices";
import { toastErrorText, toastSuccessText } from "../../helpers/toastify";


export default function Dashboards() {
  const dispatch = useDispatch();
  const auth = useSelector((state) => state.accountReducer);
  const invoicesReducer = useSelector((state) => state.invoicesReducer);
  const [filter, setFilter] = useState('All');

  useEffect(() => {
    dispatch(getInvoices());
  }, [])

  const onCompleted =  (e, InvoiceNo, CourseID) => {
    // e.target.disabled = true;
    http.post("/invoices/completed", {
      InvoiceNo: InvoiceNo,
      CourseID: CourseID, 
    }).then((result) => {
      toastSuccessText("Completed Course!");
      localStorage.setItem("time_enroll", Date.now());
      dispatch(getInvoices());
      dispatch({
        type: Types.AUTH_UPDATE,
        payload: { user: result.user }
      });
    });
  }
  const getFilterInvoices = () => {
    if (filter == 'All') {
      return invoicesReducer.data || [];
    } else if (filter == 'Completed') {
      return invoicesReducer.data.filter((item) => !!item.Completed);
    } else if (filter == 'Uncompleted') {
      return invoicesReducer.data.filter((item) => !item.Completed);
    }
  };

  return (
    <>
      {auth.isLogged ? (
        <>
          <ThemeOptions />
          <AppHeader />
          <div className="app-main ">
            <div className="app-main__inner">
              <h2>My Course</h2>
              <Row form>
                <Col md={3}>
                  <FormGroup>
                    <Label for="exampleName">Filter Course</Label>
                    <Input
                      type="select"
                      value={filter}
                      onChange={(e) => {
                        setFilter(e.target.value)
                      }}
                    >
                      <option value={"All"}>All Courses</option>
                      <option value={"Completed"}>Completed Course</option>
                      <option value={"Uncompleted"}>Uncompleted Courses</option>
                    </Input>
                  </FormGroup>
                </Col>
              </Row>
              <table className="history-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Course </th>
                    <th>Course Title </th>
                    <th>Technology Skill </th>
                    <th>Price </th>
                    <th>Purchase Date </th>
                    <th>Status</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {getFilterInvoices().map((row) => (
                    <tr key={row.InvoiceNo}>
                      <td>{row.InvoiceNo}</td>
                      <td>{row.CourseID}</td>
                      <td>{row.course.courseTitle}</td>
                      <td>{row.course.technologySkill}</td>
                      <td>{row.ItemPrice == 0
                                ? "Free"
                                : new Intl.NumberFormat("it-IT").format(
                                  row.ItemPrice) + " VNĐ"}</td>
                      <td>{row.InvoiceDate}</td>
                      <td>{!!row.Completed ? 'Completed' : 'Uncompleted'}</td>
                      <td><Button
                            disabled={!!row.Completed}
                            className="btn-wide mb-2 mr-2 btn-icon"
                            outline
                            color="primary"
                            onClick={(e) => onCompleted(e, row.InvoiceNo, row.CourseID)}
                          >Completed</Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : (
        <Redirect to="/login" />
      )}
    </>
  );
}
