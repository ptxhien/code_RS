import * as Types from "./../../constants/actionType";
import { callApi, callApiLogin } from "./../../utils/apiCaller";
import { history } from "../../../helpers/history";
import {
  toastErrorText,
  toastSuccess,
  toastSuccessText,
} from "../../../helpers/toastify";
import { API_ROUTE } from "../../constants/constant";
import queryString from "query-string";
import http from "../../utils/http";
//#region  Login
export const actionLoginRequest = args => {
  return async (dispatch) => {
    dispatch({
      type: Types.LOADING,
      payload: { },
    });
    try {
      const { authToken, user } = await http.post(API_ROUTE.URL_LOGIN, args);
      dispatch({
        type: Types.LOGIN,
        payload: {
          authToken,
          user,
        }
      });
      console.log(1);
    } catch (error) {
      
      dispatch({
        type: Types.LOGIN_FAILURE,
        payload: {
          message: error.message,
        },
      });
    }
  };
};
export const Retrieve = (args) => {
  return (dispatch) => {
    const authToken = localStorage.getItem("authToken");
    // const learnerID = localStorage.getItem("learnerID");
    // const email = localStorage.getItem("email");
    // const fullname = localStorage.getItem("fullname");
    const user = JSON.parse(localStorage.getItem("user"));

    if (authToken){
      dispatch({
        type: Types.RETRIEVE,
        payload: {
          authToken,
          user,
        }
      });
    } else {
      dispatch({type: Types.LOG_OUT})
    }
  }
  
};
//#endregion


export const Logout = (args) => {
  return (dispatch) => {
     dispatch({ type: Types.LOG_OUT });
  };
};

export const registerAction = (args) => {
  return async (dispatch) => {
    dispatch({
      type: Types.LOADING,
      payload: {},
    });
    try {
      const { authToken, user } = await http.post(API_ROUTE.URL_REGISTER, args);
      dispatch({
        type: Types.REGISTER,
        payload: {
          authToken,
          user
        }
      })
    } catch (error) {
      dispatch({
        type: Types.LOGIN_FAILURE,
        payload: {
          message: error.message,
        },
      });
    }
  };
};

export const updateAction = (args) => {
  return async (dispatch) => {
    dispatch({
      type: Types.LOADING,
      payload: {},
    });
    try {
      const { user, message } = await http.post(API_ROUTE.URL_AUTH_UPDATE, args);
      dispatch({
        type: Types.AUTH_UPDATE,
        payload: {
          user,
          message
        }
      })
    } catch (error) {
      dispatch({
        type: Types.LOGIN_FAILURE,
        payload: {
          message: error.message,
        },
      });
    }
  };
};
