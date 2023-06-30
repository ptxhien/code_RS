import React, { Fragment, useEffect, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import ThemeOptions from "../../Layout/ThemeOptions";
import AppHeader from "../../Layout/AppHeader";
import { GetJobAction } from "../../redux/masterdata/masterDataAction";
import "../Home/style.css";
import { UncontrolledCarousel } from "reactstrap";

import {
  getCourses,
  recommendCourses,
} from "../../redux/actions/courses/courses";
import { splitToSubArr } from "../../utils";

import { toastErrorText } from "../../helpers/toastify";
import { useHistory } from "react-router";
import { DefaultCourses } from "../../Components/Courses";
import RecommendationHandler from "../../Components/RecommendationHandler";
import MethodEnum from "../../Components/Courses/MethodEnum";
import RecommendationCourses from "../../Components/Courses/Recommendation";
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

const CarouselDefault = () => (
  <UncontrolledCarousel items={items} />
);
/*  */
export default function HomePage() {
  const dispatch = useDispatch();
  const { coursesReducer } = useSelector((state) => state);
  
  const [courseArrays, setCourseArrays] = useState([]);
  const [courseOnlineArrays, setCourseOnlineArrays] = useState([]);
  const [courseOfflineArrays, setCourseOfflineArrays] = useState([]);

  const [exception, setException] = useState([]);
  const [bothException, setBothException] = useState([]);
  const [bothStatus, setBothStatus] = useState(0);
  const [bothMessage, setBothMessage] = useState("");
  const [bothNgoaiLe, setBothNgoaiLe] = useState({});

  const history = useHistory();
  const itemsCountPerPage = 8;
  const [activePage, setActivePage] = useState(1);
  const [totalItemsCount, setTotalItemsCount] = useState(1);
  const [method, setMethod] = useState(MethodEnum.ONLINE);

  useEffect(() => {
    dispatch(GetJobAction());
    dispatch(getCourses());
  }, []);

  useEffect(() => {
    if (localStorage.getItem("Form") === "online") {
      setMethod(MethodEnum.ONLINE);
    } else if (localStorage.getItem("Form") === "offline") {
      setMethod(MethodEnum.OFFLINE);
    }

    localStorage.removeItem("Form");

    if (!coursesReducer.isRecommended) {
      setTotalItemsCount(coursesReducer.data.length);
      setCourseArrays(splitToSubArr(coursesReducer.data, itemsCountPerPage));
    } else {
      if (method === MethodEnum.ONLINE) {
        const onlineCourses = coursesReducer.online.Course && coursesReducer.online.Course.length
          ? coursesReducer.online.Course
          : coursesReducer.online.Ngoai_Le && coursesReducer.online.Ngoai_Le.Course_Offer &&
            coursesReducer.online.Ngoai_Le.Course_Offer.length
          ? coursesReducer.online.Ngoai_Le.Course_Offer
          : [];
        setTotalItemsCount(onlineCourses.length);
        setCourseArrays(splitToSubArr(onlineCourses, itemsCountPerPage));
        setException(coursesReducer.online && coursesReducer.online.Exception || {}); 
        setBothException(coursesReducer.online && coursesReducer.online.Exception || []);
        setBothStatus(coursesReducer.online && coursesReducer.online.status || 0);
        setBothMessage(coursesReducer.online && coursesReducer.online.message || "");
        setCourseOnlineArrays(onlineCourses || []);
        setBothNgoaiLe(coursesReducer.online && coursesReducer.online.Ngoai_Le || {});
      } else if (method === MethodEnum.OFFLINE) {
        const offlineCourses = coursesReducer.offline.Course && coursesReducer.offline.Course.length
          ? coursesReducer.offline.Course
          : coursesReducer.offline.Ngoai_Le && coursesReducer.offline.Ngoai_Le.Course_Offer &&
            coursesReducer.offline.Ngoai_Le.Course_Offer.length
          ? coursesReducer.offline.Ngoai_Le.Course_Offer
          : [];
        setTotalItemsCount(offlineCourses.length);
        setCourseArrays(splitToSubArr(offlineCourses, itemsCountPerPage));
        setException(coursesReducer.offline && coursesReducer.offline.Exception || {}); 
        setBothException(coursesReducer.offline && coursesReducer.offline.Exception || []);
        setBothStatus(coursesReducer.offline && coursesReducer.offline.status || 0);
        setBothMessage(coursesReducer.offline && coursesReducer.offline.message || "");
        setCourseOfflineArrays(offlineCourses || []);
        setBothNgoaiLe(coursesReducer.offline && coursesReducer.offline.Ngoai_Le || {});
      }
    }
  }, [coursesReducer, method]);

  const handlePageChange = (pageNumber) => {
    setActivePage(pageNumber);
  };

  return (
    <Fragment>
      <ThemeOptions />
      <AppHeader />
      <div style={{marginLeft:"auto",marginRight:"auto", marginTop: 60, padding: "20px 0", height: 250, width: "60%",}}>
      <CarouselDefault></CarouselDefault>
      </div>

      {/* < div className="app-main"> */}
      {/* <AppSidebar /> */}
      {/* <div className="app-main__outer"> */}
      <div className="app-main__inner">
        <Fragment>
          <div style={{width: "80%", margin: "auto"}}>
          <RecommendationHandler />
          </div>
          <div className="app-main__outer">
            <div className="app-main__inner mt-2 container-fluid">
              {coursesReducer.isLoading ? (
                <div className="m-auto" style={{ width: 100, height: 50 }}>
                  <img src="/images/loading.gif" style={{ width: 300, height: 300 }}></img>
                </div>
              ) : coursesReducer.isRecommended ? (
                <RecommendationCourses
                  activePage={activePage}
                  courseArrays={courseArrays}
                  courseOnlineArrays={courseOnlineArrays}
                  courseOfflineArrays={courseOfflineArrays}
                  exceptions={exception}
                  bothException={bothException}
                  bothMessage={bothMessage}
                  bothStatus={bothStatus}
                  handlePageChange={handlePageChange}
                  itemsCountPerPage={itemsCountPerPage}
                  totalItemsCount={totalItemsCount}
                  setMethod={setMethod}
                  method={method}
                  bothNgoaiLe={bothNgoaiLe}
                />
              ) : (
                <DefaultCourses
                  activePage={activePage}
                  courseArrays={courseArrays}
                  handlePageChange={handlePageChange}
                  itemsCountPerPage={itemsCountPerPage}
                  totalItemsCount={totalItemsCount}
                />
              )}
            </div>
          </div>
        </Fragment>
      </div>
    </Fragment>
  );
}
// 