import axios from "axios";
import queryString from "query-string";

const API_URL = "http://localhost:8080/api";

const http = axios.create({
  baseURL: API_URL,
  headers: {
    // "Authorization": localStorage.getItem("token") ? `Bearer ${localStorage.getItem("token")}` : undefined,
    "content-type": "application/json",
  },
  paramsSerializer: (params) => queryString.stringify(params),
});
http.interceptors.request.use(
  (config) => {
    const jwt = localStorage.getItem("authToken");
    if (jwt) {
      config.headers = {
        auth: `${jwt}`,
      };
    }
    return config;
  },
  (error) => Promise.reject(error)
);

http.interceptors.response.use(
  (response) => {
    // Return JSON data
    if (response.data) {
      return response.data;
    }
    return response;
  },
  (error) => {
    // if (error?.response?.errId) {
    //   toast.error(errorMes[error?.response?.errId]);
    // }
    // Attempt to get the actual error returned from API
    const err = (error.response && error.response.data) || error;

    return Promise.reject(err); // Propagate rejection back to caller
  }
);

export default http;
