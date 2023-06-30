import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faAngleUp, faHeart } from "@fortawesome/free-solid-svg-icons";
import {
  Col,
  Row,
  Button,
  Card,
  CardBody,
  CardTitle,
  CardSubtitle,
} from "reactstrap";
import Rating from "react-rating";
import image1 from "../../assets/images/slider-img2.jpg";
import Pagination from "react-js-pagination";
import { Link } from "react-router-dom";


function DefaultCourses({ courseArrays, activePage, itemsCountPerPage, handlePageChange, totalItemsCount}) {
  return (
    <Row>
      <Col md={12}>
        <Card>
          <CardBody>
            <Row>
              {courseArrays[activePage - 1] &&
                courseArrays[activePage - 1].map((item, index) => {
                  return (
                    <Col md="6" lg="4" xl="3" key={index}>
                      <Card
                        data-tip
                        data-for={item.courseTitle}
                        body
                        className="card-shadow-primary border mb-3 p-0"
                        outline
                        color="primary"
                        style={{ borderRadius: 12 }}
                      >
                        <CardBody title={item.courseTitle}>
                          <CardTitle
                            title={item.courseTitle}
                            className="text-truncate"
                          >
                            {item.courseTitle}
                          </CardTitle>
                          <CardSubtitle className="mb-0">
                            {item.provider}
                          </CardSubtitle>
                          <span className="mr-1 text-success">
                            {item.rating ? item.rating.toFixed(1) : 0}/5
                          </span>
                          <Rating
                            stop={5}
                            initialRating={item.rating}
                            emptySymbol={
                              <span className="mr-1 opacity-2">
                                <FontAwesomeIcon
                                  size="1x"
                                  icon={faHeart}
                                  color="red"
                                />
                              </span>
                            }
                            fullSymbol={
                              <span className="mr-1">
                                <FontAwesomeIcon
                                  size="1x"
                                  icon={faHeart}
                                  color="red"
                                />
                              </span>
                            }
                          />
                          <span className="text-info">
                            ({item.peopleRating ? item.peopleRating : 0})
                          </span>
                        </CardBody>
                        <CardBody>
                          <img src={'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRIq54GdfdVl775njXwO5XC3IjHu9IX6LzuVg&usqp=CAU'} width='{"100%"}'></img>
                          <span className="multilines-truncate">
                            {item.outcomeLearning}
                          </span>
                          <div className="d-flex justify-content-between align-items-center">
                            <Button
                              className="btn-wide mb-2 mr-2 btn-icon"
                              outline
                              color="primary"
                            >
                              <i className="pe-7s-cash btn-icon-wrapper"></i>
                              {item.feeVND == 0 ? "Free" : new Intl.NumberFormat('it-IT').format(item.feeVND) + " VNĐ"}
                            </Button>
                            <Link
                              target="_blank"
                              to={`course/${item.courseID}`}
                              className="btn-wide mb-2 btn-icon d-inline-block btn btn-outline-primary"
                            >
                              <i className="pe-7s-news-paper btn-icon-wrapper"></i>
                              Details
                            </Link>
                          </div>
                        </CardBody>
                      </Card>
                    </Col>
                  );
                })}
            </Row>
          </CardBody>
        </Card>
        <div className="my-pagination mt-2">
          <Pagination
            activePage={activePage}
            itemsCountPerPage={itemsCountPerPage}
            totalItemsCount={totalItemsCount}
            pageRangeDisplayed={5}
            onChange={handlePageChange}
          />
        </div>
      </Col>
    </Row>
  );
}

export default DefaultCourses;
