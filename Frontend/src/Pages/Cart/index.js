import React, {
  Fragment,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import ThemeOptions from "../../Layout/ThemeOptions";
import AppHeader from "../../Layout/AppHeader";
import { useCart } from "../../hooks/useCart";
import { Col, Row, Button } from "reactstrap";

export default function Cart() {
  const { cart, deleteCourse, selectItem, selectAllItem, enroll } = useCart();
  return (
    <>
      <ThemeOptions />
      <AppHeader />
      <div className="app-main__inner">
        <div
          style={{
            padding: 100,
          }}
        >
          <h1>
            <div style={{ color: "#000000" }}>
              <b>CART</b>
              <br />
            </div>
          </h1>

          {/* <h4>You have {cart.length} courses in cart</h4> */}
          <h6>
            <div style={{ color: "#0062B1" }}>
              <b>You have {cart.length} courses in cart</b>
              <br />
              &nbsp;&nbsp;&nbsp;
            </div>
          </h6>
          <div>
            <input
              id="checkall"
              type="checkbox"
              checked={!cart.filter((item) => !item.isSelected).length}
              onChange={(e) => selectAllItem(e.target.checked)}
            />
            <label style={{ marginLeft: 10 }} htmlFor="checkall">
              <br />
              ALL COURSES
              <br />
            </label>
          </div>
          <Row style={{ display: "flex" }}>
            <Col md={8} style={{ display: "flex", flexDirection: "column" }}>
              {cart.map((course, index) => (
                <Course
                  key={course.CourseID + "-" + index}
                  {...course}
                  index={index}
                  deleteCourse={deleteCourse}
                  selectItem={selectItem}
                />
              ))}
            </Col>
            <Col md={4}>
              <div
                style={{
                  backgroundColor: "lightblue",
                  padding: 20,
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                <h4>
                  <div style={{ color: "#0062B1" }}>
                    <b>CASH RECEIPT</b>
                    <br />
                    &nbsp;&nbsp;&nbsp;
                  </div>
                </h4>

                <br />
                <div
                  style={{
                    paddingLeft: 20,
                    display: "flex",
                    borderBottom: "1px solid gray",
                    justifyContent: "space-between",
                  }}
                >
                  <h5>TOTAL</h5>
                  <h5>
                    <div style={{ color: "#B80000" }}>
                      <b>
                        {Intl.NumberFormat("it-IT").format(
                          cart.reduce((acc, cur) => {
                            if (cur.isSelected) {
                              return acc + cur.feeVND;
                            }
                            return acc;
                          }, 0)
                        ) + " VNĐ"}
                      </b>
                    </div>
                  </h5>
                  <h5></h5>
                </div>
                <Button
                  onClick={enroll}
                  color="primary"
                  className=" mb-2 mt-2 ml-auto mr-2 py-2 px-4 btn-icon"
                >
                  PAY
                </Button>
              </div>
            </Col>
          </Row>
        </div>
      </div>
    </>
  );
}

function Course(props) {
  return (
    <div style={{ display: "flex", marginTop: 20 }}>
      <input
        type="checkbox"
        checked={props.isSelected || false}
        onClick={() => props.selectItem(props.index)}
      />
      <img
        style={{ marginLeft: 20 }}
        src={
          "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRIq54GdfdVl775njXwO5XC3IjHu9IX6LzuVg&usqp=CAU"
        }
      />
      <div
        style={{
          marginLeft: 20,
          display: "flex",
          flexDirection: "column",
          flex: "1",
        }}
      >
        {/* <h5>{props.courseTitle}</h5> */}
        <h5>
          <div style={{ color: "#194D33" }}>
            <b>{props.courseTitle}</b>
            <br />
            &nbsp;&nbsp;&nbsp;
          </div>
        </h5>
        <h6>
          <div style={{ color: "#009688" }}>
            <b>{props.provider}</b>
            <br />
            &nbsp;&nbsp;&nbsp;
          </div>
        </h6>

        {/* <h6>{props.provider}</h6> */}

        <h5 style={{ marginTop: "auto", marginLeft: "auto", color: "#D33115" }}>
          {props.feeVND == 0
            ? "FREE"
            : Intl.NumberFormat("it-IT").format(props.feeVND) + " VNĐ"}
        </h5>
      </div>
      <i
        onClick={() => props.deleteCourse(props.index)}
        style={{ fontSize: 36, cursor: "pointer" }}
        className="pe-7s-trash btn-icon-wrapper"
      ></i>
    </div>
  );
}
