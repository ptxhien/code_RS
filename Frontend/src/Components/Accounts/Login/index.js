import React, { Fragment } from 'react';
import { toastErrorText } from './../../../helpers/toastify';
import { library } from '@fortawesome/fontawesome-svg-core'
import { fab } from '@fortawesome/free-brands-svg-icons'
import {
    faSpinner,
} from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
library.add(
    faSpinner,
);

class LoginComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            txtUsername: '',
            txtPassword: '',
            isLoader: false
        }
    }
    onChange = (e) => {
        var target = e.target;
        var name = target.name;
        var value = target.type === 'checkbox' ? target.checked : target.value;
        this.setState({
            [name]: value
        });
    }
    onClickLogin = () => {
        let { txtUsername, txtPassword } = this.state;
        if (txtUsername == "") {
            toastErrorText("Please enter your email");
            return false;
        }
        if (txtPassword == "") {
            toastErrorText("Please enter your password");
            return false;
        }
        this.setState({
            ["isLoader"]: true
        });
        let dataSend = {
            Email: txtUsername,
            Password: txtPassword,
        }
        this.props.handlerLogin(dataSend);

        setTimeout(function () {
            this.setState({ ["isLoader"]: false });
        }.bind(this), 5000)
    }
    onKeyDown = (e) => {
        if (e.key === 'Enter') {
            this.onClickLogin();
        }
    }
    render() {
        let { txtUsername, txtPassword, isLoader } = this.state;
        let txtLoader = "";
        if (isLoader) {
            txtLoader = <FontAwesomeIcon
                icon={['fas', 'spinner']}
                pulse
                fixedWidth
                size="1x"
            />
        }
        return (
            <Fragment>
                <div className="app-container app-theme-white body-tabs-shadow">
                    <div className="app-container">
                        <div className="h-100 bg-login bg-animation">
                            <div className="d-flex h-100 justify-content-center align-items-center">
                                <div className="mx-auto app-login-box col-md-8">
                                    <div className="app-logo-inverse mx-auto mb-3"></div>
                                    <div className="modal-dialog w-100 mx-auto">
                                        <div className="modal-content">
                                            <div className="modal-body">
                                                <div className="h5 modal-title text-center">
                                                    <h4 className="mt-2">
                                                        <div>Hello,</div>
                                                        <span>Please login your account.</span>
                                                    </h4>
                                                </div>
                                                <form className="">
                                                    <div className="form-row">
                                                        <div className="col-md-12">
                                                            <div className="position-relative form-group">
                                                                <input
                                                                    placeholder="Email..."
                                                                    type="text"
                                                                    className="form-control"
                                                                    name="txtUsername"
                                                                    value={txtUsername}
                                                                    onChange={this.onChange}
                                                                    onKeyDown={this.onKeyDown}
                                                                />
                                                            </div>
                                                        </div>
                                                        <div className="col-md-12">
                                                            <div className="position-relative form-group">
                                                                <input
                                                                    placeholder="Password..."
                                                                    type="password"
                                                                    className="form-control"
                                                                    name='txtPassword'
                                                                    value={txtPassword}
                                                                    onChange={this.onChange}
                                                                    onKeyDown={this.onKeyDown}
                                                                />
                                                            </div>
                                                        </div>
                                                    </div>
                                                </form>
                                            </div>
                                            <div className="modal-footer clearfix">
                                                <div className="float-right">
                                                    <button
                                                        className="btn btn-warning btn-lg"
                                                        onClick={this.onClickLogin}
                                                        disabled={!txtUsername || !txtPassword || isLoader}
                                                    >
                                                        {txtLoader}
                                                        &nbsp;
                                                        Login
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="text-center text-white opacity-8 mt-3">Copyright © CBBANK 2020</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </Fragment>
        )
    }
}

export default LoginComponent;