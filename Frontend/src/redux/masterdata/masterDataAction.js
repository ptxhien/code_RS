import axios from "axios";
import { toastErrorText } from "../../helpers/toastify";
import * as Types from "../constants/actionType";
import { API_ROUTE } from "../constants/constant";
import { callApiNoAuthen } from "../utils/apiCaller";
import http from "../utils/http";
//#region  Login
export const GetAllLanguageAction = () => {
  return async (dispatch) => {
    dispatch({ type: Types.JOB_LOADING });
    try {
      const { data } = await http.get("/languages");
      dispatch({
        type: Types.GET_LANGUAGE,
        data,
      });
    } catch (err) {
      dispatch({
        type: Types.GET_DATA_ERROR,
      });
    }
  };
};

export const GetAllTechnologyAction = () => {
  return async (dispatch) => {
    dispatch({ type: Types.JOB_LOADING });
    try {
      const { data } = await http.get("/technologies");
      dispatch({
        type: Types.GET_TECHNOLOGY,
        data,
      });
    } catch (err) {
      dispatch({
        type: Types.GET_DATA_ERROR,
      });
    }
  };
};

export const GetFreeTimeAction = () => {
  return async (dispatch) => {
    dispatch({ type: Types.JOB_LOADING });
    try {
      const { data } = await http.get("/freetime");
      dispatch({
        type: Types.GET_FREE_TIME,
        data: data,
      });
    } catch (err) {
      dispatch({
        type: Types.GET_DATA_ERROR,
      });
    }
  };
};

export const GetJobAction = () => {
  return async (dispatch) => {
    dispatch({ type: Types.JOB_LOADING });
    try {
      const {data} = await http.get("/jobs");
      dispatch({
        type: Types.GET_JOB,
        data
      })
    } catch (err) {
      dispatch({
        type: Types.GET_DATA_ERROR,
      })
    }
    
  };
};

export const GetAllMajorAction = () => {
  return async (dispatch) => {
    dispatch({ type: Types.JOB_LOADING });
    try {
      const { data } = await http.get("/major");
      dispatch({
        type: Types.GET_MAJOR,
        data,
      });
    } catch (err) {
      dispatch({
        type: Types.GET_DATA_ERROR,
      });
    }
  };
};