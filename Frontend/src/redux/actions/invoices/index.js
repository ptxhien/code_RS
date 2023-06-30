import * as Types from "./../../constants/actionType";
import http from "../../utils/http";

export const getInvoices = () => async (dispatch) => {
  dispatch({
    type: Types.COURSE_LOADING,
  });
  try {
    const { data } = await http.get("/invoices");
    dispatch({
      type: Types.GET_INVOICE,
      payload: {
        data,
      },
    });
  } catch (err) {
    console.log(err.message);
    dispatch({
      type: Types.INVOICE_FAILED,
    });
  }
};

export const createInvoice = (CourseID, Quality, ItemPrice) => async (dispatch) => {
  dispatch({
    type: Types.COURSE_LOADING,
  });
  try {
    const { data } = await http.post("/invoices", {
      CourseID,
      Quality,
      ItemPrice,
    });
    dispatch({
      type: Types.CREATE_INVOICE,
    });
  } catch (err) {}
};

// export const showOffer
