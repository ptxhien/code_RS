import React, { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import ThemeOptions from "../../Layout/ThemeOptions";
import AppHeader from "../../Layout/AppHeader";
import "./style.scss";
import { Button, ButtonGroup, UncontrolledCarousel, Row, Col } from "reactstrap";
import avatar1 from "../../assets/utils/images/avatars/2.jpg";

import { useHistory, useParams, useLocation } from "react-router";
import http from "../../redux/utils/http";
import * as Types from "./../../redux/constants/actionType";
import { useMemo } from "react";
import { toastErrorText, toastSuccessText } from "../../helpers/toastify";
import { AiFillHome, AiOutlineLink, AiOutlineShareAlt, AiOutlineGlobal,AiOutlineClockCircle, AiFillStar, AiOutlineTeam, AiFillDollarCircle, AiFillDashboard } from "react-icons/ai";
import { useCart } from "../../hooks/useCart";
const items = [
  {
    id: 1,
    src: "https://cdn.domestika.org/c_fill,dpr_auto,f_auto,h_157,pg_1,t_base_params,w_280/v1648132526/course-covers/000/003/675/3675-original.jpg?1648132526",
    altText: "Slide 1",
    caption: "Slide 1",
  },
  {
    id: 2,
    src: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJN4eQafUM19FrQSW2YDiIFZaaZtc-ZIIEj_hJwIJIoON2hK-cnIRz5WfsfzwDyK1T-oY&usqp=CAU",
    altText: "Slide 2",
    caption: "Slide 2",
  },
  {
    id: 3,
    src: "https://images.indianexpress.com/2020/03/laptop759.jpg",
    altText: "Slide 3",
    caption: "Slide 3",
  },
  {
    id: 4,
    src: "https://d1ymz67w5raq8g.cloudfront.net/Pictures/1024x536/P/web/n/z/b/onlinecourses_shutterstock_490891228_2000px_728945.jpg",
    altText: "Slide 4",
    caption: "Slide 4",
  },
  {
    id: 5,
    src: "https://cdn.searchenginejournal.com/wp-content/uploads/2022/01/free-courses-on-content-marketing-and-writing-620b90a002530-sej-1520x800.png",
    altText: "Slide 5",
    caption: "Slide 5",
  },
  {
    id: 6,
    src: "https://www.simplilearn.com/ice9/free_resources_article_thumb/Best_Content_Writer_Salary_Across_the_World.jpg",
    altText: "Slide 6",
    caption: "Slide 6",
  },
  {
    id: 7,
    src: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSm6hRv1exWkpiGrL7oNBsmdhMQeb6glP04-rWMZjYCpL0KIuyLxgfva5KfygKq5pF0hpc&usqp=CAU",
    altText: "Slide 7",
    caption: "Slide 7",
  },
  {
    id: 8,
    src: "https://learnenglishkids.britishcouncil.org/sites/kids/files/styles/max_325x325/public/field/section/image/RS7853_ThinkstockPhotos-827490826-low.jpg?itok=JGu6j7fY",
    altText: "Slide 8",
    caption: "Slide 8",
  },
  {
    id: 9,
    src: "https://53.fs1.hubspotusercontent-na1.net/hub/53/hubfs/best-online-courses.jpg?width=595&height=400&name=best-online-courses.jpg",
    altText: "Slide 9",
    caption: "Slide 9",
  },
  {
    id: 10,
    src: "https://img-cdn.inc.com/image/upload/w_1920,h_1080,c_fill/images/panoramic/getty_1187833318_2000133220009280118_mbtvwq.jpg",
    altText: "Slide 10",
    caption: "Slide 10",
  },
];
function useQuery() {
  return new URLSearchParams(useLocation().search);
}
const CarouselDefault = (props) => (
  <UncontrolledCarousel style={{ height: "100px !important" }} items={items} />
);

const CourseDetail = () => {
  const query = useQuery()
  const skillsAcquiredArray = useMemo(() => {
    let data = query.get("skillsAcquired") || ""
    return data.split(", ")
  }, [query])

  const dispatch = useDispatch();
  const history = useHistory();
  const { id } = useParams();
  const [loading, setLoading] = useState(true);
  const [course, setCourse] = useState(false);
  const { addCourse } = useCart();
  
  const getData = useCallback(async () => {
    setLoading(true);
    try {
      const { data } = await http.get(`/courses/${id}`)
      setCourse(data);
    } catch (err) {

    }
    setLoading(false);
  }, [id]);

  useEffect(() => {
    getData();
  }, [getData]);

  const enroll = useCallback(async (e) => {
    e.target.disabled = true;
    await http.post("/invoices", {
      CourseID: course.courseID,
      Quality: 1, 
      ItemPrice: course.feeVND,
    }).then((result, a) => {
      localStorage.setItem("time_enroll", Date.now());
      history.push("/dashboard");
    }).catch((err) => {
      if (err.msg) {
        toastErrorText(err.msg);
      }
    });
  }, [course]);

  function urlshare() {
    return 'https://www.facebook.com/sharer/sharer.php?u=' + window.location;
  }
  return (
    <>
      <ThemeOptions />
      <AppHeader />
      <CarouselDefault></CarouselDefault>

      <div className="app-main__inner">
        {!course && !loading && <section id="wrapper" className="error-page my-5">
          <div className="error-box">
            <div className="error-body text-center">
              <h1>404</h1>
              <h3 className="text-uppercase">Course not found !</h3>
              <p className="text-muted m-t-30 m-b-30">Your course is currently unavailable</p>
              <a href="/" className="btn btn-info btn-rounded waves-effect waves-light m-b-40">Back to home</a> </div>
          </div>
        </section>}

        {course && <div className="container-fluid py-5">
          <div className="container py-5">
            <div className="row">
              <div className="col-lg-8">
                <div className="mb-5">
                  <h1 className="mb-5"> <b>{course.courseTitle}</b> </h1>
                  <h5><div style={{color: '#B80000'}}><b>{course.majobSubject}</b></div></h5>
                 
                  <div className="banner">
                    <img className="img-fluid rounded w-100 mb-4" src={'https://img.idesign.vn/2018/11/26/id-huong-dan-tao-bo-icon-phang-23.gif'} alt="Image" />
                    <ButtonGroup className="enroll-btn-group">

                      <Button
                        className="btn-wide btn-icon"
                        color="success"
                        onClick={() => addCourse(course)}
                      >
                        <i className="pe-7s-news-paper btn-icon-wrapper"></i>
                        Add to cart 
                      </Button>
                    </ButtonGroup>
                  </div>

                  <h6> <div style={{color: '#B80000'}}><b>WHAT YOU WILL LEARN</b><br/>&nbsp;&nbsp;&nbsp;</div></h6>

                  <p>{course.outcomeLearning}</p>
                </div>

                <div className="mb-5">
                  <h3 className="text-uppercase mb-4" style={{letterSpacing: '5px'}}>1 Comments</h3>
                  <div className="media mb-4">
                    <img src={avatar1} alt="Image" className="img-fluid rounded-circle mr-3 mt-1" style={{ width: '45px' }} />
                    <div className="media-body">
                      <h6>Phạm Thị Xuân Hiền <small><i>01 Jan 2022 at 12:00pm</i></small></h6>
                      <p>Good</p>
                      <button className="btn btn-sm btn-secondary">Reply</button>
                    </div>
                  </div>
                </div>

              </div>
              <div className="col-lg-4 mt-5 mt-lg-0">
                {/* Author Bio */}
              
                <h4><div style={{color: '#0062B1'}}><b>{course.hasOwnProperty('location') ? 'OFFLINE' : 'ONLINE'} COURSE</b><br/></div></h4>

                <div className="d-flex flex-column text-center bg-info rounded mb-5 py-5 px-4">
                  <img src={'https://png.pngtree.com/element_our/png_detail/20181226/trainingcourseonlinecomputerchat-line-icon--vector-isola-png_285274.jpg'} className="img-fluid rounded-circle mx-auto mb-3" style={{width: '100px'}} />
                  {/* <h3 className="text-white mb-3" > {course.provider}</h3> */}
                  <h3> {course.provider}</h3>
                </div>

                <div className="mb-5">
                <h4><div style={{color: '#0062B1'}}><b>INFORMATION COURSE</b>&nbsp;</div></h4>
                  <div className="d-flex flex-column rounded mb-5 px-4">
                  <Row>
                  <h7>
                  <AiFillDashboard />{course.level}&nbsp;&nbsp;
                    {course.numStudent && <div><AiOutlineTeam /> {course.numStudent}&nbsp;&nbsp;</div>}
                    <AiFillDollarCircle />{course.feeVND == 0 ? "Free" : new Intl.NumberFormat('it-IT').format(course.feeVND) + " VNĐ"}&nbsp;&nbsp;
                    {course.rating && <div><AiFillStar /> {course.rating ? course.rating.toFixed(1) : 0}&nbsp;&nbsp;</div>}
                  </h7>
                    
                  </Row>

                  <Row>
                    <div>
                    <h7>
                      <AiOutlineGlobal /> {course.language}&nbsp;&nbsp;
                      {course.studyTime && <div><AiOutlineClockCircle /> {course.studyTime}<br/></div>}
                      {course.location && <div><AiFillHome /> {course.location}</div>}
                      <div><AiOutlineLink /><a href={course.URL} target="_blank">Link: {course.URL}</a></div>
                      <a href={urlshare()} target="_blank"><AiOutlineShareAlt />Share</a>
                    </h7>
                    </div>
                  </Row>
                  </div>
                </div>
                
                {/* Tag Cloud */}
                <div className="mb-5">
           
                  <h4><div style={{color: '#0062B1'}}><b>SKILLS COURSES</b><br/></div></h4>
                  <div className="d-flex flex-wrap m-n1">
                    {course.technologySkill.split(", ").map((skill, index) => (
                      <a href="" onClick={(e) => {e.preventDefault(); console.log(Object.keys(skillsAcquiredArray))}} 
                      className={`btn ${skillsAcquiredArray.includes(skill) ? "active-btn" : "btn-outline-primary"} m-1`} key={index}>{skill}</a>
                    ))}

                  </div>

                </div>
              </div>
            </div>
          </div>
        </div>}
      </div>
    </>
  );
}

export default CourseDetail