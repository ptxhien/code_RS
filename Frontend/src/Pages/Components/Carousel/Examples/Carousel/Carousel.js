import React from "react";
import { UncontrolledCarousel } from "reactstrap";

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

const CarouselDefault = () => <UncontrolledCarousel items={items} />;

export default CarouselDefault;
