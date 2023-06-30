import axios from 'axios';
import { API_ENPOINT, API_ENPOINT_GATEWAY } from './../constants/constant';
import { history } from '../../helpers/history';
import { toastErrorText } from '../../helpers/toastify';

export var callApi = function (endpoint, method = 'GET', body) {
    let token = localStorage.getItem('token');
    return axios({
        method: method,
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        url: `${API_ENPOINT}/${endpoint}`,
        data: body
    }).catch(err => {
        console.log(err)
        if (err.response && err.response.status === 401) {
            localStorage.removeItem("token");
            localStorage.removeItem("userid");
            history.push('/');
        } else {
            return err;
        }
    });
};

export var callApiNoAuthen = function (endpoint, method = 'GET', body) {
    return axios({
        method: method,
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        },
        url: `${API_ENPOINT}/${endpoint}`,
        data: body
    }).catch(err => {
        console.log(err)
    });
};
