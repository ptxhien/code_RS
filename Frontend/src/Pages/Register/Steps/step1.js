import React, { Fragment } from "react";
import { Row, Col, FormGroup, Label, Input, Form, Button } from "reactstrap";


export default function WizardStep1({ DTO, setDTO }) {
    return (
        <Fragment>
            <div className="form-wizard-content">
                <Col lg="9" md="10" sm="12" className="mx-auto app-login-box">
                    <Form>
                        <Row form>
                            <Col md={6}>
                                <FormGroup>
                                    <Label for="exampleName"><span className="text-danger">*</span>Name</Label>
                                    <Input type="text" value={DTO.fullname} onChange={(e) => setDTO({ ...DTO, fullname: e.target.value})} />
                                </FormGroup>
                            </Col>

                            <Col md={6}>
                                <FormGroup>
                                    <Label for="exampleEmail">
                                        <span className="text-danger">*</span> Email
                                    </Label>
                                    <Input type="email" value={DTO.email} onChange={(e) => setDTO({ ...DTO, email: e.target.value.trim() })} />
                                </FormGroup>
                            </Col>

                            <Col md={6}>
                                <FormGroup>
                                    <Label for="examplePassword">
                                        <span className="text-danger">*</span> Password
                                    </Label>
                                    <Input type="password" value={DTO.password} onChange={(e) => setDTO({ ...DTO, password: e.target.value })} />
                                </FormGroup>
                            </Col>

                            <Col md={6}>
                                <FormGroup>
                                    {/* <Label for="exampleName"><span className="text-danger">*</span>Gender</Label> */}
                                    <Label for="exampleName">
                                        Gender
                                    </Label>
                                    <Input type="select" value={DTO.gender} onChange={(e) => setDTO({ ...DTO, gender: e.target.value })}>
                                        <option value={""}>Please choose your gender</option>
                                        <option value={"Male"}>Male</option>
                                        <option value={"Female"}>Female</option>
                                    </Input>
                                </FormGroup>
                            </Col>

                        </Row>
                        <div className="mt-4 d-flex align-items-center">
                            <h5 className="mb-0">
                                Already have an account?{" "}
                                <a href="/#/login" className="text-primary">
                                    Sign in
                                </a>
                            </h5>
                            {/* <div className="ml-auto">
                                <Button color="primary" className="btn-wide btn-pill btn-shadow btn-hover-shine" size="lg">
                                    Create Account
                                </Button>
                            </div> */}
                        </div>
                    </Form>
                </Col>
            </div>
        </Fragment>
    );
}
