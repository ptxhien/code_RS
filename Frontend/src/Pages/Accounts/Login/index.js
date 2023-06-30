import React from 'react';
import LoginComponent from './../../../Components/Accounts/Login';
import { connect } from 'react-redux';
import { actionLoginRequest } from './../../../redux/actions/account/accountAction';
import { useHistory } from 'react-router';
class LoginPage extends React.Component {
    constructor(props) {
        super(props);
    }
    handlerLogin = (data) => {
        this.props.login(data);
    }
    render() {
        return (
            <LoginComponent
                handlerLogin={this.handlerLogin}
            ></LoginComponent>
    )}
}


const mapStateToProps = (state) => {
    return {

    }
}
const mapDispatchToProps = (dispatch, props) => {
    return {
        login: (data) => {
            dispatch(actionLoginRequest(data))
        }
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(LoginPage);
