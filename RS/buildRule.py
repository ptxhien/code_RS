import function
import knowledgeDomain
import pandas as pd
# -*- coding: utf-8 -*-


def convertlst_toString(lst):
    new_lst = []
    [new_lst.append(i) for i,v in lst.items()]
    return ", ".join(new_lst)

# build  online


def BuildRule_Online(df_On, missing_skill, lan_know, occupation_id, feeMax, 
                     condition_duration, typeFilter):
    flat_level = 0
    flat_language = 0
    
    dict_f_ngoaile = []
    dict_f = []

    df_Off = []
    lan_no_know = []
    
    result = pd.DataFrame()
    result_ngoaile = pd.DataFrame()
    
    kq_result = []
    kq_result_ngoaile = []

    rule_On_lan = pd.DataFrame()
    rule_On_remain = pd.DataFrame()
    
    # job similarity
    lst_job_sim = knowledgeDomain.job_related(occupation_id)
    del lst_job_sim[0:1]
    str_lst_job_sim = ", ".join(lst_job_sim)


    if len(df_On) > 0:
        rule_On_lan, flat_language = knowledgeDomain.Xet_Language(
            df_On, df_Off, "online", lan_know)
        if len(rule_On_lan) > 0:
            rule_On_lan = rule_On_lan.loc[(rule_On_lan.level == 'Beginner') | (
                rule_On_lan.level == 'ALL Levels')]
            
            if len(rule_On_lan) > 0:
                rule_On_lan = function.similar_bert(rule_On_lan, occupation_id)
                if typeFilter.lower() == "top":
                    rule_On_lan = function.Course_Weight_Top_BERT(
                        rule_On_lan, "online")
                else:
                    rule_On_lan = function.Course_Weight_BERT(
                        rule_On_lan, occupation_id, "online")
                result = rule_On_lan
                
            else:
                flat_level = -1

        else:
            lan_no_know = function.Find_Language_Remaining_LearnNotKnow(
                df_On, lan_know)
            str_lan_no_know = ", ".join(lan_no_know)

            rule_On_remain, flat_language = knowledgeDomain.Xet_Language(
                df_On, df_Off, "online", lan_no_know)
            
            if len(rule_On_remain) > 0:
                rule_On_remain = rule_On_remain.loc[(rule_On_remain.level == 'Beginner') | (
                    rule_On_remain.level == 'ALL Levels')]

                if len(rule_On_remain) > 0:
                    rule_On_remain = function.similar_bert(
                    rule_On_remain, occupation_id)
                    
                    if typeFilter.lower() == "top":
                        rule_On_remain = function.Course_Weight_Top_BERT(
                            rule_On_remain, "online")
                        
                    else:
                        rule_On_remain = function.Course_Weight_BERT(
                            rule_On_remain, occupation_id, "online")
                    
                    result_ngoaile = rule_On_remain
                    flat_language = -1
                else:
                    flat_language = -1
                    flat_level = -1

        for i, r in result.iterrows():
            kq_result.append({"courseID": str(r[0]),
                              "courseTitle": str(r[2]),
                              "Tech_Skill": str(r[4]),
                              "studyTime": "",
                              "studyForm": "",
                              "technologySkill": str(r[3]),
                              "outcomeLearning": str(r[9]),
                              "provider": str(r[11]),
                              "duration": str(r[12]),
                              "feeVND": str(r[15]),
                              "URL": str(r[10]),
                              "language": str(r[20]),
                              "rating": str(r[17]),
                              "peopleRating": str(r[18]),
                              "location": "",
                              "level": str(r[14]),
                            #   'distance': "",
                            #   'Weight':str(r[25]),
                              "is_online": "true"})


        for i, r in result_ngoaile.iterrows():
            kq_result_ngoaile.append({"courseID": str(r[0]),
                                      "courseTitle": str(r[2]),
                                      "Tech_Skill": str(r[4]),
                                      "studyTime": "",
                                      "studyForm": "",
                                      "technologySkill": str(r[3]),
                                      "outcomeLearning": str(r[9]),
                                      "provider": str(r[11]),
                                      "duration": str(r[12]),
                                      "feeVND": str(r[15]),
                                      "URL": str(r[10]),
                                      "language": str(r[20]),
                                      "rating": str(r[17]),
                                      "peopleRating": str(r[18]),
                                      "location": "",
                                      "level": str(r[14]),
                                    #   'distance': "",
                                    #   'Weight':str(r[25]),
                                      "is_online": "true"})

        # TH1: dont courses language and level
        if (flat_language == -1 and flat_level == -1) or (flat_language == 0 and flat_level == -1):
            ExceptionType = "Lan, Level"
            dict_f_ngoaile.append({"ExceptionType": ExceptionType,
                                   "Job_offer": str_lst_job_sim})

            dict_f = {"status": 401,
                      "message": "Lan, Level",
                      "Course": [],
                      "Exception": dict_f_ngoaile,
                      "Ngoai_Le": {
                          "Course_Offer": [],
                          "ExceptionDetail": []
                      }}

        # TH2: courses have other languages to use
        elif flat_language == -1 and flat_level == 0:
            lstSkill_Provider_ngoaile, lstSkill_notProvider_ngoaile = function.lst_Skill_RS(
                result_ngoaile, missing_skill, occupation_id)
            dict_f_ngoaile.append({"ExceptionType": "Lan",
                                   "lan_remain": str_lan_no_know})

            str_new_lstSkill_Provider_ngoaile = convertlst_toString(
                lstSkill_Provider_ngoaile)
            str_new_lstSkill_notProvider_ngoaile = ", ".join(
                lstSkill_notProvider_ngoaile)

            # --------------------
            dict_f_ngoaile.append({"lstSkill_Provider_ngoaile": str_new_lstSkill_Provider_ngoaile,
                                   "lstSkill_notProvider_ngoaile": str_new_lstSkill_notProvider_ngoaile})

            # Fee and Duration for the learning route
            if typeFilter.lower() != "top":
                if feeMax != "":
                    flat_sum_fee_RS, fee_Learner, sum_fee_course, sum_bothem = knowledgeDomain.TinhSumFeeRS(
                        result_ngoaile, feeMax)
                    nguong_max = function.convertfee(fee_Learner)
                    if flat_sum_fee_RS == -1:
                        dict_f_ngoaile.append({"ExceptionType": "Fee",
                                               "Input": str(nguong_max),
                                               "Output": str(sum_fee_course),
                                               "Balance": str(sum_bothem)})

                if condition_duration > 0:
                    flat_sum_duration, sum_learn_duration, sum_course_duration, kq_hocthem = knowledgeDomain.TinhSumDurationRS(
                        result_ngoaile, condition_duration)
                    if flat_sum_duration == -1:
                        dict_f_ngoaile.append({"ExceptionType": "Duration",
                                               "Input": str(sum_learn_duration),
                                               "Output": str(sum_course_duration),
                                               "Balance": str(kq_hocthem)})

            dict_f = {"status": 402,
                      "message": "Lan",
                      "Course": [],
                      "Exception": [],
                      "Ngoai_Le": {"Course_Offer": kq_result_ngoaile,
                                   "ExceptionDetail": dict_f_ngoaile}}

        # TH3: COURSE HAS THE RIGHT LANGUAGE AND LEVEL
        elif flat_language == 0 and flat_level == 0:
            lstSkill_Provider, lstSkill_notProvider = function.lst_Skill_RS(
                result, missing_skill, occupation_id)
            str_new_lstSkill_Provider = convertlst_toString(lstSkill_Provider)
            str_new_lstSkill_notProvider = ", ".join(lstSkill_notProvider)
            # str_new_lstSkill_notProvider = convertlst_toString(lstSkill_notProvider)

            dict_f_ngoaile.append({"lstSkill_Provider": str_new_lstSkill_Provider,
                                   "lstSkill_notProvider": str_new_lstSkill_notProvider})

            if typeFilter.lower() != "top":
                if feeMax != "":
                    flat_sum_fee_RS, fee_Learner, sum_fee_course, sum_bothem = knowledgeDomain.TinhSumFeeRS(
                        result, feeMax)
                    nguong_max = function.convertfee(fee_Learner)

                    if flat_sum_fee_RS == -1:
                        dict_f_ngoaile.append({"ExceptionType": "Fee",
                                               "Input": str(nguong_max),
                                               "Output": str(sum_fee_course),
                                               "Balance": str(sum_bothem)})

                if condition_duration > 0:
                    flat_sum_duration, sum_learn_duration, sum_course_duration, kq_hocthem = knowledgeDomain.TinhSumDurationRS(
                        result, condition_duration)
                    if flat_sum_duration == -1:
                        dict_f_ngoaile.append({"ExceptionType": "Duration",
                                               "Input": str(sum_learn_duration),
                                               "Output": str(sum_course_duration),
                                               "Balance": str(kq_hocthem)})

            dict_f = {"status": 200,
                      "message": "PASS",
                      "Course": kq_result,
                      "Exception": dict_f_ngoaile,
                      "Ngoai_Le": {"Course_Offer": [],
                                   "ExceptionDetail": []}}

    else:
        result = df_On
        dict_f_ngoaile.append({"Job_offer": str_lst_job_sim})

        dict_f = {"status": 403,
                  "message": "FAIL",
                  "Course": [],
                  "Exception": dict_f_ngoaile,
                  "Ngoai_Le": {
                      "Course_Offer": [],
                      "ExceptionDetail": []}}

    return result, dict_f

# build  offline


def Test_Location_FreeTime_JobNow(result, lat1, lon1, Learner_Job_Now, Learner_FreeTime):
    flat_course_freetime = 0

    # 2. Location
    if lat1 != "" and lon1 != "":
        result = knowledgeDomain.Xet_Location(result, lat1, lon1)

    # 3. Frame time and Job now
    if len(result) > 0:
        if Learner_FreeTime == "":
            df, flat_course_freetime = knowledgeDomain.Xet_FrameStudy_JobNow(
                result, Learner_Job_Now, '18:00-23:00')
            if len(df) > 0:
                result = df

        else:

            df, flat_course_freetime = knowledgeDomain.Xet_FrameStudy_JobNow(
                result, Learner_Job_Now, Learner_FreeTime)
            if len(df) > 0:
                result = df

    return result, flat_course_freetime


def Test_Duration_Fee(result, condition_duration, feeMax, typeFilter):
    dict_f_ngoaile = []

    if len(result) > 0:
        if typeFilter.lower() != "top":
            if feeMax != "":
                flat_sum_fee_RS, fee_Learner, sum_fee_course, sum_bothem = knowledgeDomain.TinhSumFeeRS(
                    result, feeMax)
                nguong_max = function.convertfee(fee_Learner)
                if flat_sum_fee_RS == -1:
                    dict_f_ngoaile.append({"ExceptionType": "Fee",
                                           "Input": str(nguong_max),
                                           "Output": str(sum_fee_course),
                                           "Balance": str(sum_bothem)})

            if condition_duration > 0:
                flat_sum_duration, sum_learn_duration, sum_course_duration, kq_hocthem = knowledgeDomain.TinhSumDurationRS(
                    result, condition_duration)
                if flat_sum_duration == -1:
                    dict_f_ngoaile.append({"ExceptionType": "Duration",
                                           "Input": str(sum_learn_duration),
                                           "Output": str(sum_course_duration),
                                           "Balance": str(kq_hocthem)})

    return result, dict_f_ngoaile


def Off_Lan(result, missing_skill, occupation, feeMax, condition_duration, 
            lat1, lon1, Learner_Job_Now, Learner_FreeTime, typeFilter):
    dict_f_Offline = {}
    dict_f_ngoaile = []
    kq_result = []
    freetime_remain = []

    result, flat_course_freetime = Test_Location_FreeTime_JobNow(
        result, lat1, lon1, Learner_Job_Now, Learner_FreeTime)

    # Courses available in a different timeframe
    if flat_course_freetime == 1:
        freetime_remain = function.Find_List_FrameTime_Remain(
            result, Learner_FreeTime)
        str_freetime_remain = ", ".join(freetime_remain)

        ExceptionType = "Frame_Remain"
        dict_f_ngoaile.append({"ExceptionType": ExceptionType,
                               "frame_remain": str_freetime_remain})
    # result
    lstSkill_Provider, lstSkill_notProvider = function.lst_Skill_RS(
        result, missing_skill, occupation)
    str_new_lstSkill_Provider = convertlst_toString(lstSkill_Provider)
    str_new_lstSkill_notProvider = ", ".join(lstSkill_notProvider)
    if len(result) > 0:
        for i, r in result.iterrows():
            kq_result.append({"courseID": str(r[0]),
                              "courseTitle": str(r[7]),
                              "Tech_Skill": str(r[2]),
                              "studyTime": str(r[17]),
                              "studyForm": str(r[13]),
                              "technologySkill": str(r[18]),
                              "outcomeLearning": str(r[9]),
                              "provider": str(r[12]),
                              "duration": str(r[14]),
                              "feeVND": str(r[16]),
                              "URL": str(r[8]),
                              "language": str(r[20]),
                              "rating": "",
                              "peopleRating": "",
                              "location": str(r[11]),
                              "level": str(r[19]),
                            #   "distance": str(r[26]),
                            #   'Weight':str(r[25]),
                              "is_online": "false"})

        result, dict_f_ngoaile_W = Test_Duration_Fee(
            result, condition_duration, feeMax, typeFilter)
        if len(result) > 0:
            for i in dict_f_ngoaile_W:
                dict_f_ngoaile.append(i)

        dict_f_ngoaile.append({"lstSkill_Provider_ngoaile": str_new_lstSkill_Provider,
                               "lstSkill_notProvider_ngoaile": str_new_lstSkill_notProvider})

    dict_f_Offline = {"status": 201,
                      "message": "frameRemain_Fulltime",
                      "Course": kq_result,
                      "Exception": dict_f_ngoaile,
                      "Ngoai_Le": {"Course_Offer": [],
                                   "ExceptionDetail": []}}
    return result, dict_f_Offline


def Off_NotLan(result, missing_skill, lan_no_know, occupation, feeMax, 
               condition_duration, lat1, lon1, Learner_Job_Now, 
               Learner_FreeTime, typeFilter):
    dict_f_Offline = {}
    dict_f_ngoaile = []
    kq_result = []
    freetime_remain = []

    result, flat_course_freetime = Test_Location_FreeTime_JobNow(
        result, lat1, lon1, Learner_Job_Now, Learner_FreeTime)

    # Part-time course available but in a different time frame
    if flat_course_freetime == 1:
        freetime_remain = function.Find_List_FrameTime_Remain(
            result, Learner_FreeTime)
        str_freetime_remain = ", ".join(freetime_remain)

        ExceptionType = "Frame_Remain"
        dict_f_ngoaile.append({"ExceptionType": ExceptionType,
                               "frame_remain": str_freetime_remain})

    lstSkill_Provider, lstSkill_notProvider = function.lst_Skill_RS(
        result, missing_skill, occupation)
    str_new_lstSkill_Provider = convertlst_toString(lstSkill_Provider)
    str_new_lstSkill_notProvider = ", ".join(lstSkill_notProvider)

    if len(result) > 0:
        for i, r in result.iterrows():
            kq_result.append({"courseID": str(r[0]),
                              "courseTitle": str(r[7]),
                              "Tech_Skill": str(r[2]),
                              "studyTime": str(r[17]),
                              "studyForm": str(r[13]),
                              "technologySkill": str(r[18]),
                              "outcomeLearning": str(r[9]),
                              "provider": str(r[12]),
                              "duration": str(r[14]),
                              "feeVND": str(r[16]),
                              "URL": str(r[8]),
                              "language": str(r[20]),
                              "rating": "",
                              "peopleRating": "",
                              "location": str(r[11]),
                              "level": str(r[19]),
                            #   "distance": str(r[26]),
                            #   'Weight':str(r[25]),
                              "is_online": "false"})

        str_lan_no_know = ", ".join(lan_no_know)
        dict_f_ngoaile.append({"ExceptionType": 'Lan',
                               "lan_remain": str_lan_no_know})

        result, dict_f_ngoaile_W = Test_Duration_Fee(
            result, condition_duration, feeMax, typeFilter)
        if len(result) > 0:
            for i in dict_f_ngoaile_W:
                dict_f_ngoaile.append(i)

        dict_f_ngoaile.append({"lstSkill_Provider_ngoaile": str_new_lstSkill_Provider,
                               "lstSkill_notProvider_ngoaile": str_new_lstSkill_notProvider})

    dict_f_Offline = {"status": 202,
                      "message": "frameRemain_Fulltime",
                      "Course": [],
                      "Exception": [],
                      "Ngoai_Le": {"Course_Offer": kq_result,
                                   "ExceptionDetail": dict_f_ngoaile}}

    return result, dict_f_Offline


def BuildRule_Offline(df_Off, missing_skill, lan_know, lat1, lon1, occupation_id, Learner_Job_Now,
                      Learner_FreeTime, feeMax, condition_duration, typeFilter):
    flat_level = 0
    flat_language = 0
    
    dict_f_Offline = {}
    dict_f_ngoaile = []
    
    df_On = []
    lan_no_know = []
    lan_no_know_copy = []
    
    result = pd.DataFrame()
    
    rule_Off_lan = pd.DataFrame()
    rule_Off_remain = pd.DataFrame()
    
    # job similarity
    lst_job_sim = knowledgeDomain.job_related(occupation_id)
    del lst_job_sim[0:1]
    str_lst_job_sim = ", ".join(lst_job_sim)
    # ----------------------------------------------------------------

    if len(df_Off) > 0:
        rule_Off_lan, flat_language = knowledgeDomain.Xet_Language(
            df_On, df_Off, "offline", lan_know)
        if len(rule_Off_lan) > 0:
            rule_Off_lan = rule_Off_lan.loc[(rule_Off_lan.level == 'Beginner') | (
                rule_Off_lan.level == 'ALL Levels')]

            if len(rule_Off_lan) > 0:
                rule_Off_lan = function.similar_bert(rule_Off_lan, occupation_id)
                if typeFilter.lower() == "top":
                    rule_Off_lan = function.Course_Weight_Top_BERT(
                        rule_Off_lan, "offline")
                    
                else:
                    rule_Off_lan = function.Course_Weight_BERT(
                        rule_Off_lan, occupation_id, "offline")
                
                result = rule_Off_lan
            
            else:
                flat_level = -1

        else:
            lan_no_know = function.Find_Language_Remaining_LearnNotKnow(
                df_Off, lan_know)
            str_lan_no_know = ", ".join(lan_no_know)
            rule_Off_remain, flat_language = knowledgeDomain.Xet_Language(
                df_On, df_Off, "offline", lan_no_know)
            if len(rule_Off_remain) > 0:
                rule_Off_remain = rule_Off_remain.loc[(rule_Off_remain.level == 'Beginner') | (
                    rule_Off_remain.level == 'ALL Levels')]
                
                if len(rule_Off_remain) > 0:
                    rule_Off_remain = function.similar_bert(
                    rule_Off_remain, occupation_id)
                    
                    if typeFilter.lower() == "top":
                        rule_Off_remain = function.Course_Weight_Top_BERT(
                            rule_Off_remain, "offline")
                       
                    else:
                        rule_Off_remain = function.Course_Weight_BERT(
                            rule_Off_remain, occupation_id, "offline")
                    
                    result = rule_Off_remain
                    
                    flat_language = -1
                else:
                    flat_language = -1
                    flat_level = -1

        lan_no_know_copy = lan_no_know
        
        # TH1: COURSE IS NOT COMPATIBLE FOR BOTH LANGUAGE AND LEVEL
        if (flat_language == -1 and flat_level == -1) or (flat_language == 0 
                                                          and flat_level == -1):
            ExceptionType = "Lan, Level"
            dict_f_ngoaile.append({"ExceptionType": ExceptionType,
                                   "Job_offer": str_lst_job_sim})

            dict_f_Offline = {"status": 401,
                              "message": "Lan, Level",
                              "Course": [],
                              "Exception": dict_f_ngoaile,
                              "Ngoai_Le": {
                                  "Course_Offer": [],
                                  "ExceptionDetail": []}}

        # TH2: COURSE IS NOT LANGUAGE RIGHT
        elif flat_language == -1 and flat_level == 0:
            if len(result) > 0:
                result, dict_f_Offline = Off_NotLan(result, missing_skill, lan_no_know_copy, occupation_id,
                                                    feeMax, condition_duration, lat1, lon1,
                                                    Learner_Job_Now, Learner_FreeTime, typeFilter)

        # TH3: LANGUAGE AND LEVEL FITS COURSE
        elif flat_language == 0 and flat_level == 0:
            result, dict_f_Offline = Off_Lan(result, missing_skill, occupation_id, feeMax,
                                             condition_duration, lat1, lon1, Learner_Job_Now,
                                             Learner_FreeTime, typeFilter)
            


    else:
        result = df_Off
        dict_f_ngoaile.append({"Job_offer": str_lst_job_sim})

        dict_f_Offline = {"status": 403,
                          "message": "FAIL",
                          "Course": [],
                          "Exception": dict_f_ngoaile,
                          "Ngoai_Le": {
                              "Course_Offer": [],
                              "ExceptionDetail": []}}
    return result, dict_f_Offline

# Check Online + Offline

def KiemTraOnlineNgoaiLe(df_On, missing_skill, lan_know, occupation_id, feeMax, condition_duration, typeFilter,
                         ):
    result_Online, dict_f_Online = BuildRule_Online(df_On, missing_skill, lan_know, occupation_id, feeMax, 
                     condition_duration, typeFilter)
    return result_Online, dict_f_Online


def KiemTraOfflineNgoaiLe(df_Off, missing_skill, lan_know, lat1, lon1, occupation, Learner_Job_Now, Learner_FreeTime, feeMax, 
                          condition_duration, typeFilter):
    result_Offline, dict_f_Offline = BuildRule_Offline(
        df_Off, missing_skill, lan_know, lat1, lon1, occupation, Learner_Job_Now, Learner_FreeTime, 
        feeMax, condition_duration, typeFilter)
    return result_Offline, dict_f_Offline

# RECOMMENDATION SYSTEMS


def recommendation(df_On, df_Off, df_attribute_requirement):
    dict_f_Offline = {}
    dict_f_ngoaile = []
    dict_f_ngoaile1 = []
    dict_f_online = {}
    dict_f = {}

    df_rule = pd.DataFrame()
    result_offline = pd.DataFrame()
    result_online = pd.DataFrame()

    # ----------------
    typeFilter = df_attribute_requirement.typeFilter[0]
    typeFilter_Name = function.typeFilterName(typeFilter)

    str_skills_to_learn = function.FindMissingSkill1(df_attribute_requirement)
    missing_skill = function.FindMissingSkill(df_attribute_requirement)
       
    occupation_id = df_attribute_requirement.Occupation[0]
    skills_acquired, occupation_title, d_knowledge = function.Find_Require_Job(occupation_id)

    lan_know = df_attribute_requirement.language[0].split(', ')
    lat1 = df_attribute_requirement.latitude[0]
    lon1 = df_attribute_requirement.longitude[0]
    Form_require = df_attribute_requirement.Form_require[0].lower()

    Learner_Job_Now = df_attribute_requirement.jobNow[0]

    Learner_FreeTime = df_attribute_requirement.freeTime[0]
    feeMax = df_attribute_requirement.feeMax[0]
    condition_duration = df_attribute_requirement.durationSecond[0]
    

    # -----------------
    lst_job_sim = knowledgeDomain.job_related(occupation_id)
    del lst_job_sim[0:1]
    str_lst_job_sim = ", ".join(lst_job_sim)
    # -----------------

    if not Form_require:
        print("Don't choose form")
        if Learner_Job_Now.startswith('work') or Learner_Job_Now.startswith('study'):
            result_online, dict_f_online = BuildRule_Online(
                df_On, missing_skill, lan_know, occupation_id,
                feeMax, condition_duration, typeFilter)
            if len(result_online) > 0:
                dict_f_ngoaile1.append({"job_offer": str_lst_job_sim})
                dict_f_ngoaile = {
                    'occupation': occupation_title,
                    'typeFilter': typeFilter_Name,
                    'skills_acquired': skills_acquired,
                    'skills_to_learn': str_skills_to_learn,
                    'courses_online': dict_f_online,
                    'courses_offline': {
                        "status": 400,
                        "message": "no courses",
                        "Course": [],
                        "Exception": dict_f_ngoaile1,
                        "Ngoai_Le": {
                            "Course_Offer": [],
                            "ExceptionDetail": []}}}
            else:
                result_offline, kq_On = KiemTraOfflineNgoaiLe(
                    df_Off, missing_skill, lan_know, lat1, lon1, occupation_id, 
                    Learner_Job_Now, Learner_FreeTime, feeMax, condition_duration, typeFilter)
                dict_f_ngoaile = {
                    'occupation': occupation_title,
                    'typeFilter': typeFilter_Name,
                    'skills_acquired': skills_acquired,
                    'skills_to_learn': str_skills_to_learn,
                    'courses_online': dict_f_online,
                    'courses_offline': kq_On}
        else:
            result_online, dict_f_online = BuildRule_Online(
                df_On, missing_skill, lan_know, occupation_id, feeMax, 
                condition_duration, typeFilter)
            
            result_offline, dict_f_Offline = BuildRule_Offline(
                df_Off, missing_skill, lan_know, lat1, lon1, occupation_id, 
                Learner_Job_Now, 
                Learner_FreeTime, feeMax, condition_duration, typeFilter)
         

            # -----
            if len(result_online) > 0 and len(result_offline) > 0:
                dict_f_ngoaile = {
                    'occupation': occupation_title,
                    'typeFilter': typeFilter_Name,
                    'skills_acquired': skills_acquired,
                    'skills_to_learn': str_skills_to_learn,
                    'courses_online': dict_f_online,
                    'courses_offline': dict_f_Offline}

                df_rule = pd.concat([result_online, result_offline])

            elif len(result_online) == 0 and len(result_offline) > 0:
                df_rule = result_offline
                dict_f_ngoaile = {
                    'occupation': occupation_title,
                    'typeFilter': typeFilter_Name,
                    'skills_acquired': skills_acquired,
                    'skills_to_learn': str_skills_to_learn,
                    'courses_online': dict_f_online,
                    'courses_offline': dict_f_Offline}

            elif len(result_online) > 0 and len(result_offline) == 0:
                df_rule = result_online
                dict_f_ngoaile = {
                    'occupation': occupation_title,
                    'typeFilter': typeFilter_Name,
                    'skills_acquired': skills_acquired,
                    'skills_to_learn': str_skills_to_learn,
                    'courses_online': dict_f_online,
                    'courses_offline': dict_f_Offline}

            elif len(result_online) == 0 and len(result_offline) == 0:
                df_rule = pd.concat([result_online, result_offline])
                dict_f_ngoaile = {
                    'occupation': occupation_title,
                    'typeFilter': typeFilter_Name,
                    'skills_acquired': skills_acquired,
                    'skills_to_learn': str_skills_to_learn,
                    'courses_online': dict_f_online,
                    'courses_offline': dict_f_Offline}
        dict_f = dict_f_ngoaile

    elif Form_require.startswith('online'):
        print("Choose Online")
        df_rule, dict_onl = BuildRule_Online(
            df_On, missing_skill, lan_know, occupation_id, feeMax, 
            condition_duration, typeFilter)
        # ----
        dict_f_ngoaile1.append({"Job_offer": str_lst_job_sim})
        # ----
        dict_f_ngoaile = {
            'occupation': occupation_title,
            'typeFilter': typeFilter_Name,
            'skills_acquired': skills_acquired,
            'skills_to_learn': str_skills_to_learn,
            'courses_online': dict_onl,
            'courses_offline': {
                "status": 400,
                "message": "no courses",
                "Course": [],
                "Exception": dict_f_ngoaile1,
                "Ngoai_Le": {
                    "Course_Offer": [],
                    "ExceptionDetail": []}}
        }
        if len(df_rule) == 0:
            print("Don't have result online. Therefore rs offline")
            result_Offline, kq_On = KiemTraOfflineNgoaiLe(
                df_Off, missing_skill, lan_know, lat1, lon1, occupation_id,
                Learner_Job_Now, Learner_FreeTime, feeMax, 
                condition_duration, typeFilter)
            dict_f_ngoaile = {
                'occupation': occupation_title,
                'typeFilter': typeFilter_Name,
                'skills_acquired': skills_acquired,
                'skills_to_learn': str_skills_to_learn,
                'courses_online': dict_onl,
                'courses_offline': kq_On}

        dict_f = dict_f_ngoaile

    else:
        print("Choose Offline")
        df_rule, dict_off = BuildRule_Offline(df_Off, missing_skill, lan_know, lat1, lon1,
                                              occupation_id, Learner_Job_Now, 
                                              Learner_FreeTime, feeMax, 
                                              condition_duration, typeFilter)
        # ----
        dict_f_ngoaile1.append({"Job_offer": str_lst_job_sim})
        # ----
        dict_f_ngoaile = {
            'occupation': occupation_title,
            'typeFilter': typeFilter_Name,
            'skills_acquired': skills_acquired,
            'skills_to_learn': str_skills_to_learn,
            'courses_online': {
                "status": 400,
                "message": "no courses",
                "Course": [],
                "Exception": dict_f_ngoaile1,
                "Ngoai_Le": {
                    "Course_Offer": [],
                    "ExceptionDetail": []}},
            'courses_offline': dict_off}

        if len(df_rule) == 0:
            result_Offline, kq_Off = KiemTraOnlineNgoaiLe(
                df_On, missing_skill, lan_know, occupation_id, 
                feeMax, condition_duration, typeFilter)
            dict_f_ngoaile = {
                'occupation': occupation_title,
                'typeFilter': typeFilter_Name,
                'skills_acquired': skills_acquired,
                'skills_to_learn': str_skills_to_learn,
                'courses_online': kq_Off,
                'courses_offline': dict_off}

        dict_f = dict_f_ngoaile

    return dict_f
