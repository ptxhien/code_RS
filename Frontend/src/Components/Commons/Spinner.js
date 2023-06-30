import React from 'react'
import Loader from "react-loader-advanced";
import { Loader as LoaderAnim } from "react-loaders";

export default function Spinner() {
    return (
        <div><h1 className="font-weight-bold text-info">Đang tải dữ liệu...<LoaderAnim color="black" type="line-scale" /></h1></div>
    )
}
