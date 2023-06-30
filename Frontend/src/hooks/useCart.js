import React, { useEffect, useContext, createContext } from "react";
import { useLocalStorage } from "./useLocalStorage";
import { useHistory } from "react-router";
import http from "../redux/utils/http";
import { toastErrorText, toastSuccessText } from "../helpers/toastify";

const cartContext = createContext();

export function ProvideCart({ children }) {
  const cart = useProvideCart();
  return <cartContext.Provider value={cart}>{children}</cartContext.Provider>;
}

export const useCart = () => {
  return useContext(cartContext);
};

function useProvideCart() {
  const [cart, setCart] = useLocalStorage("cart", []);
  const history = useHistory();
  useEffect(() => {}, []);
  const addCourse = (course) => {
    let findItem = cart.find((item) => item.courseID == course.courseID);
    if (findItem) {
        toastErrorText('Course bought');
    } else {
      http.post("/invoices/check-exist", {
        CourseID: course.courseID,
      }).then((result, a) => {
        // localStorage.setItem("time_enroll", Date.now());
        setCart([...cart, course]);
      }).catch((err) => {
        if (err.msg) {
          toastErrorText(err.msg);
        }
      });
    }
  };
  const deleteCourse = (index) => {
    setCart(cart.filter((_, idx) => index != idx));
  };

  const selectItem = (index) => {
    setCart(
      cart.map((item, idx) => ({
        ...item,
        isSelected: index == idx ? !item.isSelected : item.isSelected,
      }))
    );
  };
  const selectAllItem = (check) => {
    setCart(cart.map((item) => ({ ...item, isSelected: check })));
  };
  const enroll = async () => {
    await Promise.all(
      cart
        .filter((course) => course.isSelected)
        .map((course, idx) =>
          new Promise(resolve => setTimeout(resolve, 500 * idx)).then(() => http.post("/invoices", {
            CourseID: course.courseID,
            Quality: 1,
            ItemPrice: course.feeVND,
          }))
        )
    );
    setCart(cart.filter((item) => !item.isSelected));
    history.push("/dashboard");
  };
  return {
    cart,
    addCourse,
    deleteCourse,
    selectItem,
    selectAllItem,
    enroll,
  };
}
