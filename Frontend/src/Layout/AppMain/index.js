import { Route, Redirect } from "react-router-dom";
import React, { Suspense, lazy, Fragment } from "react";
import Loader from "react-loaders";
import routes from './../../router';
import { PrivateRoute } from './PrivateRoute';
import { ToastContainer } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';

const Dashboard = lazy(() => import("../../Pages/Dashboards"));

const AppMain = () => {
    return (
        <Fragment>
            {showMenu(routes)}
            <Suspense fallback={
                <div className="loader-container">
                    <div className="loader-container-inner">
                        <div className="text-center">
                            <Loader type="ball-pulse-rise" />
                        </div>
                        <h6 className="mt-5">
                            Please wait while we load all the Forms examples
                            <small>Because this is a demonstration we load at once all the Forms examples. This wouldn't happen in a real live app!</small>
                        </h6>
                    </div>
                </div>
            }>
                <Route path="/Dashboard" component={Dashboard} />
            </Suspense>

            <ToastContainer />
        </Fragment>
    )
};


const showMenu = (routes) => {
    let result = null;
    if (routes.length > 0) {
        result = routes.map((item, index) => {
            if (item.auth) {
                return (
                    <Suspense key={index} fallback={
                        <div className="loader-container">
                            <div className="loader-container-inner">
                                <div className="text-center">
                                    <Loader type="ball-pulse-rise" />
                                </div>
                                <h6 className="mt-5">
                                    Please wait while we load all the Forms examples
                                    <small>Because this is a demonstration we load at once all the Forms examples. This wouldn't happen in a real live app!</small>
                                </h6>
                            </div>
                        </div>
                    }>
                        <PrivateRoute
                            key={index}
                            path={item.path}
                            exact={item.exact}
                            component={item.main}
                        />
                    </Suspense>
                );
            }
            else {
                return (
                    <Suspense key={index} fallback={
                        <div className="loader-container">
                            <div className="loader-container-inner">
                                <div className="text-center">
                                    <Loader type="ball-pulse-rise" />
                                </div>
                                <h6 className="mt-5">
                                    Please wait while we load all the Forms examples
                                    <small>Because this is a demonstration we load at once all the Forms examples. This wouldn't happen in a real live app!</small>
                                </h6>
                            </div>
                        </div>
                    }>
                        <Route
                            key={index}
                            path={item.path}
                            exact={item.exact}
                            component={item.main}
                        >
                        </Route>
                    </Suspense>

                )
            }
        })
    }
    return result;
}

export default AppMain;