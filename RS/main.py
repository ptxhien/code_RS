from flask import Flask, request
# jsonify, render_template
from flask_cors import CORS, cross_origin
import buildRule
import dao
import function
import knowledgeDomain
import pandas as pd

# -*- coding: utf-8 -*-

# start flask server backend
app = Flask(__name__)

# apply Flask Cors
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# ------------- RS -------------#
@app.route("/recommendation", methods=["POST", "GET"])
@cross_origin(origin='*')
def RS():
    # 1. lay thong tin learner chon tren fontend
    occupation = request.args.get("occupation").strip()
    occupation_Name = function.Find_Title(occupation)

    form = request.args.get("form")
    month = request.args.get("month")
    email = request.args.get("email")
    typeFilter = request.args.get("typeFilter")
    typeFilter_Name = function.typeFilter_Name(typeFilter)

    # 2. load information relate learner
    # user profile
    df_attribute_requirement = dao.User_Preq_Attributes(
        email, occupation, form, month, typeFilter)

    # load courses
    df_On = function.take_CourseOnline(df_attribute_requirement)
    df_Off = function.take_CourseOffline(df_attribute_requirement)

    # # find missing skill
    str_skills_acquired = function.Find_Skill_Weight(
        df_attribute_requirement.Occupation[0])
    str_skills_to_learn = function.FindMissingSkill_1(df_attribute_requirement)

    # ----------------------------------------------------------------
    lst_job_sim = knowledgeDomain.job_related(occupation)
    del lst_job_sim[0:1]
    str_lst_job_sim = ", ".join(lst_job_sim)
    # ----------------------------------------------------------------

    dict_f = {}

    # # 3. Dua vao model
    if len(df_attribute_requirement) > 0:
        dict_f_ngoaile1 = []

        if len(str_skills_to_learn) > 0:
            if len(df_On) > 0 or len(df_Off) > 0:
                dict_f = buildRule.recommendation(
                    df_On, df_Off, df_attribute_requirement, str_skills_acquired, str_skills_to_learn, occupation_Name, typeFilter_Name)
            else:
                dict_f_ngoaile1.append({"Job_offer": str_lst_job_sim})
                dict_f = {
                    'occupation': occupation_Name,
                    'typeFilter': typeFilter_Name,
                    'skills_acquired': str_skills_acquired,
                    'skills_to_learn': str_skills_to_learn,
                    'courses_offline': {
                        "status": 400,
                        "message": "no courses",
                        "Course": [],
                        "Exception": dict_f_ngoaile1,
                        "Ngoai_Le": {
                            "Course_Offer": [],
                            "ExceptionDetail": []}},
                    'courses_online': {
                        "status": 400,
                        "message": "no courses",
                        "Course": [],
                        "Exception": dict_f_ngoaile1,
                        "Ngoai_Le": {
                            "Course_Offer": [],
                            "ExceptionDetail": []}}}
        else:
            dict_f = {
                'occupation': occupation_Name,
                'typeFilter': typeFilter_Name,
                'skills_acquired': str_skills_acquired,
                'skills_to_learn': str_skills_to_learn,
                'courses_offline': {
                    "status": 203,
                    "message": "enough skills",
                    "Course": [],
                    "Exception": [],
                    "Ngoai_Le": {
                        "Course_Offer": [],
                        "ExceptionDetail": []}},
                'courses_online': {
                    "status": 203,
                    "message": "enough skills",
                    "Course": [],
                    "Exception": [],
                    "Ngoai_Le": {
                        "Course_Offer": [],
                        "ExceptionDetail": []}}}

    else:
        dict_f = {"message": "This user doesn't exist",
                  "status": 407}

    # save file result
    # file_name = 'KQ_RS.json'
    # with open(file_name, 'w') as f:
    #     json.dump(dict_f, f, ensure_ascii=False, indent=4)

    # 4. tra ket qua ve trang chu hien thi
    # return render_template("index.html", dict_f)
    return dict_f


# start backend
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port='6868')
