from curses import KEY_A1
import json
import pandas as pd
import dao
import numpy
from difflib import SequenceMatcher
from math import *
import json
from math import cos, asin, sqrt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# 1. Find courses studied


def Find_Courses_Studied(email):
    conn = dao.create_connection()
    df_Learner = dao.select_l(conn)
    df_Learner = df_Learner[df_Learner['email'] == email]
    df_Learner = df_Learner.reset_index(drop=True)
    learner_id = df_Learner['learnerID'][0]

    df_Invoice = dao.select_invoice(conn)

    # df_Invoice['TotalPrice'] = df_Invoice['Quality'] * df_Invoice['ItemPrice']
    df_loc_his = df_Invoice.copy()

    df_loc_his = df_loc_his.loc[df_loc_his.LearnerID == learner_id]
    lst_courses_studied = []
    for i, r in df_loc_his.iterrows():
        if r['CourseID'] not in lst_courses_studied:
            lst_courses_studied.append(r['CourseID'])
    lst_courses_studied.sort()
    return lst_courses_studied

# 2.2 Find Skills Job Requirement


def read_rule_job():
    f = open('Rule_Job_h.json',)
    data = json.loads(f.read())
    return data


def Find_Skill_Weight(occupation_id):
    data = read_rule_job()
    d_skill = {}
    for i in data:
        if i['JobID'] == occupation_id:
            d_skill = i['Weight_Technology']
    d_skill = (dict(sorted(d_skill.items(), key=lambda x: x[1], reverse=True)))
    return d_skill


def Find_Title(occupation_id):
    data = read_rule_job()
    for i in data:
        if i['JobID'] == occupation_id:
            return i['JobTitle']


def Find_Know(occupation_id):
    data = read_rule_job()
    for i in data:
        if i['JobID'] == occupation_id:
            return i['knowledge']

# 3. Find Skill User Missing


def FindMissingSkill_1(df_attribute_requirement):
    occupation = df_attribute_requirement.Occupation[0]
    d_skill = Find_Skill_Weight(occupation)
    lst_weight_sort = d_skill

    skill_now_learner = []
    for id, row in df_attribute_requirement.iterrows():
        for tec in row.loc['technologySkill'].split(', '):
            if tec != '' and tec not in skill_now_learner:
                skill_now_learner.append(tec)
    skill_now_learner.sort()

    missing_skill = lst_weight_sort
    common_l = list(set(skill_now_learner) & set(missing_skill))
    if len(common_l) > 0:
        for i in common_l:
            missing_skill.pop(i)
    return missing_skill


def FindMissingSkill(df_attribute_requirement):
    occupation = df_attribute_requirement.Occupation[0]
    d_skill = Find_Skill_Weight(occupation)
    lst_weight_sort = sorted(d_skill, key=d_skill.get, reverse=True)

    skill_now_learner = []
    for id, row in df_attribute_requirement.iterrows():
        for tec in row.loc['technologySkill'].split(', '):
            if tec != '' and tec not in skill_now_learner:
                skill_now_learner.append(tec)
    skill_now_learner.sort()

    missing_skill = lst_weight_sort
    common_l = list(set(skill_now_learner) & set(missing_skill))
    if len(common_l) > 0:
        for i in common_l:
            missing_skill.remove(i)
    return missing_skill

# 4. Find courses contains skills user missing


def FindCourseFromMissingSkill(df, df_attribute_requirement):
    missing_skill = FindMissingSkill(df_attribute_requirement)

    df_Course_Filter = pd.DataFrame()

    if len(missing_skill) > 0:
        TEMP = pd.get_dummies(missing_skill, dtype=int)
        df_Course_Filter = df[['courseID', 'courseTitle', 'technologySkill']]
        df_Course_Filter = df_Course_Filter.join(TEMP)
        df_Course_Filter = df_Course_Filter.astype(str)

        skill = df_Course_Filter.columns[3:].tolist()

        for id, row in df_Course_Filter.iterrows():
            for k in range(len(skill)):
                df_Course_Filter.at[id, skill[k]] = '0'

        for id, row in df_Course_Filter.iterrows():
            for tec in row.loc['technologySkill'].split(', '):
                if (tec != ''):
                    for k in range(len(skill)):
                        if (tec.lower().strip() != skill[k].lower().strip()):
                            continue
                        else:
                            df_Course_Filter.at[id, skill[k]] = '1'

        count_row = df_Course_Filter.shape[0]
        for i in range(count_row):
            skillset = ""
            for k in range(len(skill)):
                b = int(df_Course_Filter.at[i, skill[k]])
                if b > 0:
                    if skillset == "":
                        skillset = skill[k]
                    else:
                        skillset = skillset + ", " + skill[k]
            df_Course_Filter.at[i, 'Tech_Skill'] = skillset

        skills = df_Course_Filter["Tech_Skill"]
        skills = skills.str.split(", ")
        for i, s in enumerate(skills):
            if ('' not in s):
                df_Course_Filter.at[i, 'Num_Skill'] = len(s)
            else:
                df_Course_Filter.at[i, 'Num_Skill'] = '0'

        df_Course_Filter.Num_Skill = df_Course_Filter.Num_Skill.apply(
            lambda x: int(x))
        df_Course_Filter = df_Course_Filter.loc[df_Course_Filter['Num_Skill'] > 0]

        occupation = df_attribute_requirement.Occupation[0]
        d_skill = Find_Skill_Weight(occupation)
        for id, row in df_Course_Filter.iterrows():
            sum_weight = 0
            for tec in row.loc['Tech_Skill'].split(', '):
                if (tec != ''):
                    for key, value in d_skill.items():
                        if tec.lower().strip() == key.lower().strip():
                            sum_weight += value
                        else:
                            continue
            df_Course_Filter.at[id, 'Sum_Weight'] = sum_weight

            lst_A = []
            lst_B = []
            for tec in row.loc['technologySkill'].split(', '):
                if tec != '' and tec not in lst_A:
                    lst_A.append(tec)

            for tec in row.loc['Tech_Skill'].split(', '):
                if tec != '' and tec not in lst_B:
                    lst_B.append(tec)

            for i in lst_B:
                for j in lst_A:
                    if j in i:
                        lst_A.remove(j)
            df_Course_Filter.at[id, 'Tech_Remain'] = ", ".join(lst_A)

        for id, row in df_Course_Filter.iterrows():
            dem = []
            for tec in row.loc['Tech_Remain'].split(', '):
                if tec != '' and tec not in dem:
                    dem.append(tec)
            df_Course_Filter.at[id, 'Num_Tech_Remain'] = len(dem)

    return df_Course_Filter



# 4. Take courses
# Online


def take_CourseOnline(df_attribute_requirement):
    email = df_attribute_requirement.email[0]

    conn = dao.create_connection()
    df_Course_Online = dao.select_courseOnline(conn)
    df_Online = df_Course_Online.copy()
    df_courses_On = pd.DataFrame()
    df_loc_On = FindCourseFromMissingSkill(
        df_Course_Online, df_attribute_requirement)

    if len(df_loc_On) > 0:
        df_loc_On = df_loc_On[['courseID', 'courseTitle', 'technologySkill',
                               'Tech_Skill', 'Num_Skill', 'Sum_Weight', 'Tech_Remain', 'Num_Tech_Remain']]
        lst_courses_studied = Find_Courses_Studied(email)
        if len(lst_courses_studied) > 0:
            for i in lst_courses_studied:
                df_loc_On = df_loc_On.drop(
                    df_loc_On[df_loc_On['courseID'] == i].index)

        df_Online = df_Online[['courseID', 'outcomeLearning', 'URL', 'provider', 'duration', 'durationSecond',
                               'level', 'feeVND', 'majobSubject', 'rating', 'peopleRating', 'numStudent', 'language']]
        df_courses_On = pd.merge(df_loc_On, df_Online,
                                 how='left', on='courseID')
        df_courses_On.feeVND = df_courses_On.feeVND.apply(lambda x: float(x))
    return df_courses_On

# offline


def take_CourseOffline(df_attribute_requirement):
    email = df_attribute_requirement.email[0]

    conn = dao.create_connection()
    df_Course_Offline = dao.select_courseOffline(conn)

    df_Offline = df_Course_Offline.copy()
    df_courses_Off = pd.DataFrame()

    df_loc_Off = FindCourseFromMissingSkill(
        df_Course_Offline, df_attribute_requirement)

    if len(df_loc_Off) > 0:
        df_loc_Off = df_loc_Off[['courseID', 'Tech_Skill',
                                 'Num_Skill', 'Sum_Weight', 'Tech_Remain', 'Num_Tech_Remain']]

        # remove the courses the learner has taken
        lst_courses_studied = Find_Courses_Studied(email)
        if len(lst_courses_studied) > 0:
            for i in lst_courses_studied:
                df_loc_Off = df_loc_Off.drop(
                    df_loc_Off[df_loc_Off['courseID'] == i].index)

        df_courses_Off = pd.merge(
            df_loc_Off, df_Offline, how='left', on='courseID')
        df_courses_Off = df_courses_Off.fillna('')

    return df_courses_Off

# 15. find similar bert courses - knowledgeDomain


def similar_bert(df_courses_On, occupation_id):
    # get embedding all courses
    bert = SentenceTransformer("all-mpnet-base-v2")

    df_courses_On['out'] = df_courses_On['outcomeLearning'] + ' '+ df_courses_On['majobSubject']
    sentence_embeddings_df = bert.encode(df_courses_On['out'].tolist())
    # get embedding job
    know = []
    know.append(Find_Know(occupation_id))
    sentence_embeddings_job = bert.encode(know)

    similarity = cosine_similarity(
        sentence_embeddings_job, sentence_embeddings_df)

    df_courses_On['index'] = [i for i in range(0, len(df_courses_On))]
    df_1 = pd.DataFrame(similarity.tolist())
    df_1_1 = df_1.T
    df_1_1 = df_1_1.rename(columns={0: "cosin_bert"})
    df_1_1['index'] = [i for i in range(0, len(df_1_1))]
    new_data = pd.merge(df_courses_On, df_1_1, how='left', on="index")
    new_data = new_data.drop(columns="index")
    new_data = new_data.sort_values(by='cosin_bert', ascending=False)
    return new_data
# 5. Find languages user know


def Language_Learner_Know(df_attribute_requirement):
    lst_lan_know = []
    for id, row in df_attribute_requirement.iterrows():
        for lan in row.loc['language'].split(', '):
            if (lan != '' and lan not in lst_lan_know):
                lst_lan_know.append(lan)
    return lst_lan_know

# 6. Find languages remain in courses

def Find_Language_Remaining_LearnNotKnow(df, lst_lan_know):
    lst_lan_know = list(lst_lan_know)
    lst_allLan = []
    for id, row in df.iterrows():
        for lan in row.loc['language'].split(', '):
            if lan != '':
                if lan.strip() not in lst_allLan and lan.strip() not in lst_lan_know:
                    lst_allLan.append(lan)
    lst_allLan.sort()
    return lst_allLan

# 7.  find courses by language


def findCourseOn_basedOn_Language(df, lst_lan):
    df_Online = df.copy()

    TEMP = pd.get_dummies(lst_lan, dtype=int)
    df_Filter = df[['courseID', 'courseTitle', 'language']]
    df_Filter = df_Filter.join(TEMP)

    lan_1 = df_Filter.columns[3:].tolist()
    df_Filter = df_Filter.astype(str)
    for id, row in df_Filter.iterrows():
        for k in range(len(lan_1)):
            df_Filter.at[id, lan_1[k]] = '0'

    for id, row in df_Filter.iterrows():
        for tec in row.loc['language'].split(', '):
            if (tec != ''):
                for k in range(len(lan_1)):
                    if (tec.strip() != lan_1[k].strip()):
                        continue
                    else:
                        df_Filter.at[id, lan_1[k]] = '1'

    df_Filter = df_Filter.astype(str)
    count_row = df_Filter.shape[0]
    for i in range(count_row):
        lanset = ""
        for k in range(len(lan_1)):
            b = int(df_Filter.at[i, lan_1[k]])
            if b > 0:
                if lanset == "":
                    lanset = lan_1[k]
                else:
                    lanset = lanset + ", " + lan_1[k]
        df_Filter.at[i, 'Sum_Lan'] = lanset

    skills = df_Filter["Sum_Lan"]
    skills = skills.str.split(", ")
    for i, s in enumerate(skills):
        if ('' not in s):
            df_Filter.at[i, 'Num_Lan'] = len(s)
        else:
            df_Filter.at[i, 'Num_Lan'] = '0'

    df_Filter.Num_Lan = df_Filter.Num_Lan.apply(lambda x: int(x))
    df_Filter = df_Filter.loc[df_Filter['Num_Lan'] > 0]

    df_Filter = df_Filter[['courseID', 'Sum_Lan']]
    df_Filter = pd.merge(df_Filter, df_Online, how='left', on='courseID')

    return df_Filter

# 8. Find frame time where offline courses are open


def Find_List_FrameTime(df):
    lst_StudyTime = []
    for id, row in df.iterrows():
        for t in row.loc['studyTime'].split('|'):
            if t != '' and t not in lst_StudyTime:
                lst_StudyTime.append(t)
    lst_StudyTime.sort()
    return lst_StudyTime


def Find_List_FrameTime_Remain(df, t_learner):
    lst_StudyTime = Find_List_FrameTime(df)
    lst_StudyTime = list(set(lst_StudyTime))

    lst_t_learner = []
    for i in t_learner.split(', '):
        lst_t_learner.append(i)
    lst_t_learner = list(set(lst_t_learner))

    lst_StudyTime = list(set(lst_StudyTime) - set(lst_t_learner))
    return lst_StudyTime

# 9.  find the skills that RS advises


def LstTech(df):
    lst_T = []
    for id, row in df.iterrows():
        for tec in row.loc['Tech_Skill'].split(', '):
            if tec != '' and tec not in lst_T:
                lst_T.append(tec)
    return lst_T


def LstTechCourse_Provider(df, occupation):
    d_F = {}
    d_skill = Find_Skill_Weight(occupation)
    lst_T = LstTech(df)

    for j in lst_T:
        for key, value in d_skill.items():
            if j == key:
                d_F.setdefault(key, value)
    return d_F


def LstTechCourse_NotProvider(lst, missing_skill):
    d_not_F = missing_skill
    for key1, value1 in lst.items():
        for key in missing_skill:
            if key == key1:
                d_not_F.remove(key1)
    return d_not_F


def lst_Skill_RS(df, missing_skill, occupation):
    lstSkill_Provider = LstTechCourse_Provider(df, occupation)
    lstSkill_notProvider = LstTechCourse_NotProvider(
        lstSkill_Provider, missing_skill)
    return lstSkill_Provider, lstSkill_notProvider

# 10. Compare Fee


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def convertfee(fee_Learner):
    nguong_max = 0
    fee_Learner = str(fee_Learner)
    if similar('0', fee_Learner):
        nguong_max = 5000000
    elif similar('1', fee_Learner):
        nguong_max = 15000000
    elif similar('2', fee_Learner):
        nguong_max = 30000000
    elif similar('3', fee_Learner):
        nguong_max = 100000000
    return float(nguong_max)


def FindFeeLess(df, feeMax):
    flat_fee = 0
    nguong_max = convertfee(feeMax)
    test_fee = df.loc[(df.feeVND >= 0) | (df.feeVND <= nguong_max)]
    if len(test_fee) > 0:
        df = test_fee
    else:
        flat_fee = -1
    return df, flat_fee

# 11. Recommend courses according to the top that provide the most skills


def standardize(row):
    new_row = abs((row - row.min()) / (row.max()-row.min()))
    # new_row = round(new_row, 2)
    return new_row


def Course_Weight_Top_BERT(df_RS, filter):
    anpha = 0.4
    belta = 0.4
    df_Course_RS = pd.DataFrame()

    df_RS = df_RS.sort_values(by='Num_Skill', ascending=False)
    if len(df_RS) > 1:
        df_RS["Sum_Weight_Stand"] = standardize(df_RS[['Sum_Weight']])
        df_RS["Num_Tech_Remain_Stand"] = standardize(
            df_RS[['Num_Tech_Remain']])

        df_RS["Weight"] = anpha * df_RS["Sum_Weight_Stand"] + belta * \
            df_RS["cosin_bert"] + (1 - anpha - belta) * \
            df_RS["Num_Tech_Remain_Stand"]

        if filter == "online":
            df_RS = df_RS.sort_values(
                ['Weight', 'numStudent', 'rating'], ascending=[False, False, False])
        else:
            df_RS = df_RS.sort_values(['Weight'], ascending=[False])

        df_Course_RS = df_RS.head(10)
    else:
        df_Course_RS = df_RS.head(10)
    return df_Course_RS


# def Course_Weight_Top(df_RS, filter):
    anpha = 0.5
    df_Course_RS = pd.DataFrame()

    df_RS = df_RS.sort_values(by='Num_Skill', ascending=False)
    if len(df_RS) > 1:
        df_RS["Sum_Weight_Stand"] = standardize(df_RS[['Sum_Weight']])
        df_RS["Num_Tech_Remain_Stand"] = standardize(
            df_RS[['Num_Tech_Remain']])
        df_RS["Weight1"] = df_RS["Sum_Weight"] + df_RS["Num_Tech_Remain"]
        df_RS["Weight"] = anpha * df_RS["Sum_Weight_Stand"] + \
            (1 - anpha) * df_RS["Num_Tech_Remain_Stand"]

        if filter == "online":
            df_RS = df_RS.sort_values(
                ['Weight', 'numStudent', 'rating'], ascending=[False, False, False])
        else:
            df_RS = df_RS.sort_values(['Weight'], ascending=[False])

        df_Course_RS = df_RS.head(10)
    else:
        df_Course_RS = df_RS.head(10)
    return df_Course_RS

# 12. Recommend courses according learning path


def Course_Weight_BERT(rule_On, occupation, filter):
    Missing_Skill = LstTechCourse_Provider(rule_On, occupation)
    anpha = 0.4
    belta = 0.4

    max_Sum_Weight = rule_On['Sum_Weight'].max()
    min_Sum_Weight = rule_On['Sum_Weight'].min()

    # ===
    RS_Skill = []
    Course_RS = []
    Course_Update = []
    df_Course_RS = pd.DataFrame()
    df_Update = pd.DataFrame()
    df_RS = rule_On.copy()

    # -----
    if len(df_RS) > 0:
        df_RS["Sum_Weight_Stand"] = standardize(df_RS[['Sum_Weight']])
        df_RS["Num_Tech_Remain_Stand"] = standardize(
            df_RS[['Num_Tech_Remain']])
        df_RS["Weight"] = anpha * df_RS["Sum_Weight_Stand"] + belta * \
            df_RS["cosin_bert"] + (1 - anpha - belta) * \
            df_RS["Num_Tech_Remain_Stand"]

        for index, row in df_RS.iterrows():
            if filter.lower() == "online":
                df_RS = df_RS.sort_values(
                    ['Weight', 'numStudent', 'rating'], ascending=[False, False, False])
            else:
                df_RS = df_RS.sort_values(['Weight'], ascending=[False])

            kq = set(row["Tech_Skill"].split(", ")) & set(RS_Skill)

            if len(kq) == 0:
                if len(df_Update) == 0:
                    Course_RS.append(row)
                    df_Course_RS = pd.DataFrame(Course_RS)
                    for l in row["Tech_Skill"].split(", "):
                        RS_Skill.append(l)
                    for i in RS_Skill:
                        Missing_Skill.pop(i, None)

                    df_RS = df_RS.drop(
                        df_RS[df_RS['courseID'] == row.courseID].index)

                else:
                    for index_update, row_update in df_Update.iterrows():
                        df_Update = df_Update.sort_values(
                            ['Weight'], ascending=[False])

                        kq_update = set(row["Tech_Skill"].split(", ")) & set(
                            row_update["Tech_Skill"].split(", "))

                        if len(kq_update) > 0:
                            if row_update.Weight == row.Weight and row.courseID != row_update.courseID:
                                if row_update.provider != row.provider:
                                    Course_RS.append(row_update)
                                    df_Course_RS = pd.DataFrame(Course_RS)

                                    for l in row.Tech_Skill.split(", "):
                                        RS_Skill.append(l)
                                    for i in RS_Skill:
                                        Missing_Skill.pop(i, None)

                                    df_RS = df_RS.drop(
                                        df_RS[df_RS['courseID'] == row.courseID].index)
                                    df_Update = df_Update.drop(
                                        df_Update[df_Update['courseID'] == row_update.courseID].index)

                            elif row_update.Weight > row.Weight and row.courseID != row_update.courseID:
                                # print("lớn trọng số", row.courseID, row_update.courseID, row.Weight, row_update.Weight, row.Tech_Skill,  row_update.Tech_Skill)

                                Course_RS.append(row_update)
                                df_Course_RS = pd.DataFrame(Course_RS)

                                for l in row_update.Tech_Skill.split(', '):
                                    RS_Skill.append(l)
                                for i in RS_Skill:
                                    Missing_Skill.pop(i, None)

                                df_Course_RS = df_Course_RS.drop(
                                    df_Course_RS[df_Course_RS['courseID'] == row.courseID].index)
                                df_RS = df_RS.drop(
                                    df_RS[df_RS['courseID'] == row.courseID].index)
                                df_Update = df_Update.drop(
                                    df_Update[df_Update['courseID'] == row_update.courseID].index)

                        else:
                            t1 = row.Tech_Skill.split(', ')
                            t2 = row_update.Tech_Skill.split(', ')

                            df_RS_t1 = set(t1) & set(RS_Skill)
                            df_update_t2 = set(t2) & set(RS_Skill)

                            if len(df_update_t2) == 0:
                                subDataFrame = df_Course_RS.loc[df_Course_RS['courseID']
                                                                == row_update.courseID]
                                if len(subDataFrame) == 0:
                                    Course_RS.append(row_update)
                                    df_Course_RS = pd.DataFrame(Course_RS)

                                for l in t2:
                                    RS_Skill.append(l)
                                for i in RS_Skill:
                                    Missing_Skill.pop(i, None)

                                df_Update = df_Update.drop(
                                    df_Update[df_Update['courseID'] == row_update.courseID].index)

                            elif len(df_RS_t1) == 0:
                                subDataFrame = df_Course_RS.loc[df_Course_RS['courseID']
                                                                == row.courseID]
                                if len(subDataFrame) == 0:
                                    Course_RS.append(row)
                                df_Course_RS = pd.DataFrame(Course_RS)

                                for l in t1:
                                    RS_Skill.append(l)
                                for i in RS_Skill:
                                    Missing_Skill.pop(i, None)

                                df_RS = df_RS.drop(
                                    df_RS[df_RS['courseID'] == row.courseID].index)

            else:
                tech_trung = row.Tech_Skill.split(', ')
                for i in RS_Skill:
                    for j in tech_trung:
                        if i == j:
                            tech_trung.remove(j)

                d_skill = Find_Skill_Weight(occupation)
                w_tech_trung = {}
                for j in tech_trung:
                    for key, value in d_skill.items():
                        if j == key:
                            w_tech_trung.setdefault(key, value)

                if len(w_tech_trung) == 0:

                    for index1, r in df_Course_RS.iterrows():
                        if r.Weight == row.Weight and r.courseID != row.courseID:
                            if r.provider != row.provider:
                                subDataFrame = df_Course_RS.loc[df_Course_RS['courseID']
                                                                == row.courseID]
                                if len(subDataFrame) == 0:
                                    Course_RS.append(row)
                                df_Course_RS = pd.DataFrame(Course_RS)

                    df_RS = df_RS.drop(
                        df_RS[df_RS['courseID'] == row.courseID].index)
                else:
                    num_skill_new = len(w_tech_trung)

                    sum_weight_new = 0
                    for key, value in w_tech_trung.items():
                        sum_weight_new = sum_weight_new + value

                    w_tech_trung = ", ".join([str(char)
                                             for char in w_tech_trung])

                    row.loc['Num_Skill'] = num_skill_new
                    row.loc['Tech_Skill'] = w_tech_trung
                    row.loc['Sum_Weight'] = sum_weight_new

                    update_Sum_Weight_Stand = abs(
                        (sum_weight_new - min_Sum_Weight)/(max_Sum_Weight - min_Sum_Weight))
                    row.loc['Sum_Weight_Stand'] = update_Sum_Weight_Stand
                    row.loc["Weight"] = anpha * update_Sum_Weight_Stand + belta * \
                        row["cosin_bert"] + (1 - anpha - belta) * \
                        row["Num_Tech_Remain_Stand"]

                    Course_Update.append(row)
                    df_Update = pd.DataFrame(Course_Update)

                    df_RS = df_RS.drop(
                        df_RS[df_RS['courseID'] == row.courseID].index)

        df_Course_RS = df_Course_RS.sort_values(['Weight'], ascending=[False])
    else:
        df_Course_RS = df_RS

    return df_Course_RS


# def Course_Weight(rule_On, occupation, filter):
#     Missing_Skill = LstTechCourse_Provider(rule_On, occupation)
#     anpha = 0.5

#     RS_Skill = []
#     Course_RS = []
#     Course_Update = []
#     df_RS = rule_On.copy()
#     df_Course_RS = pd.DataFrame()

#     maxNumSkill = df_RS["Num_Skill"].max()
#     df_RS = df_RS.sort_values(by='Num_Skill', ascending=False)
#     if len(df_RS) > 1:
#         while maxNumSkill > 0:
#             df_RS["Sum_Weight_Stand"] = standardize(df_RS[['Sum_Weight']])
#             df_RS["Num_Tech_Remain_Stand"] = standardize(
#                 df_RS[['Num_Tech_Remain']])

#             df_RS["Weight1"] = df_RS["Sum_Weight"] + df_RS["Num_Tech_Remain"]
#             df_RS["Weight"] = anpha * df_RS["Sum_Weight_Stand"] + \
#                 (1 - anpha) * df_RS["Num_Tech_Remain_Stand"]

#             rows = df_RS.loc[df_RS['Num_Skill'] == maxNumSkill]

#             if filter.lower() == "online":
#                 rows = rows.sort_values(['Weight', 'numStudent', 'rating'], ascending=[
#                                         False, False, False])
#             else:
#                 rows = rows.sort_values(['Weight'], ascending=[False])

#             for index, row in rows.iterrows():
#                 lst_skill = row["Tech_Skill"].split(", ")
#                 kq = set(lst_skill) & set(RS_Skill)

#                 if len(kq) == 0:
#                     Course_RS.append(row)
#                     df_Course_RS = pd.DataFrame(Course_RS)
#                     for l in lst_skill:
#                         RS_Skill.append(l)
#                     for i in RS_Skill:
#                         Missing_Skill.pop(i, None)
#                 else:
#                     tech_trung = row.Tech_Skill.split(', ')

#                     for i in RS_Skill:
#                         for j in tech_trung:
#                             if i == j:
#                                 tech_trung.remove(j)

#                     d_skill = Find_Skill_Weight(occupation)
#                     w_tech_trung = {}
#                     for j in tech_trung:
#                         for key, value in d_skill.items():
#                             if j == key:
#                                 w_tech_trung.setdefault(key, value)

#                     if len(w_tech_trung) == 0:
#                         for index1, r in df_Course_RS.iterrows():
#                             if r.Weight == row.Weight and r.courseID != row.courseID:
#                                 if r.provider != row.provider:
#                                     subDataFrame = df_Course_RS.loc[df_Course_RS['courseID']
#                                                                     == row.courseID]
#                                     if len(subDataFrame) == 0:
#                                         Course_RS.append(row)
#                                     df_Course_RS = pd.DataFrame(Course_RS)
#                     else:
#                         num_skill_new = len(w_tech_trung)
#                         # skill_remain = row.Num_Tech_Remain
#                         sum_weight_new = 0

#                         for key, value in w_tech_trung.items():
#                             sum_weight_new = sum_weight_new + value

#                         w_tech_trung = "".join([str(char)
#                                                for char in w_tech_trung])

#                         row.loc['Num_Skill'] = num_skill_new
#                         row.loc['Sum_Weight'] = sum_weight_new
#                         row.loc['Tech_Skill'] = w_tech_trung

#                         Course_Update.append(row)
#                         df_Update = pd.DataFrame(Course_Update)
#                         df_RS = df_RS.drop(
#                             df_RS[df_RS['courseID'] == row.courseID].index)
#                         df_RS = pd.concat([df_RS, df_Update])

#             df_Course_RS = pd.DataFrame(Course_RS)
#             maxNumSkill -= 1
#     else:
#         df_Course_RS = df_RS
#     return df_Course_RS

# 13. Find offline courses based on learners' free time frames

def get_frame_days(t):
    lst_day = []
    f_time = t[:11]

    f = t[12:]
    f = f[1:]
    f = f[:-1]
    lst_day = f.split('-')

    return f_time, lst_day


def FindCoursebasedStudyTime(df, t_learner):
    lst_df1 = []
    dem = 0
    dem_khong = 0
    # -----
    f_time, lst_day = get_frame_days(t_learner)
    learn_h_start = f_time[0:2]
    learn_h_start = pd.to_numeric(learn_h_start, downcast='integer')
    learn_s_start = f_time[3:5]
    learn_s_start = pd.to_numeric(learn_s_start, downcast='integer')
    learn_h_end = f_time[6:8]
    learn_h_end = pd.to_numeric(learn_h_end, downcast='integer')
    learn_s_end = f_time[9:11]
    learn_s_end = pd.to_numeric(learn_s_end, downcast='integer')
    learn_frame = t_learner[12:]
    # -----

    for id, row in df.iterrows():
        for sTime in row.loc['studyTime'].split('|'):
            if sTime != '':
                sTime_h_start = sTime[0:2]
                sTime_h_start = pd.to_numeric(
                    sTime_h_start, downcast='integer')
                sTime_s_start = sTime[3:5]
                sTime_s_start = pd.to_numeric(
                    sTime_s_start, downcast='integer')
                sTime_h_end = sTime[6:8]
                sTime_h_end = pd.to_numeric(sTime_h_end, downcast='integer')
                sTime_s_end = sTime[9:11]
                sTime_s_end = pd.to_numeric(sTime_s_end, downcast='integer')
                f_time_c, lst_day_c = get_frame_days(sTime)

                if (sTime_h_start >= learn_h_start and sTime_h_end <= learn_h_end) and (sTime_s_start >= learn_s_start and sTime_s_end <= learn_s_end):
                    if learn_frame != "":
                        if sTime[12:] == t_learner[12:]:
                            lst_df1.append(row)
                        else:
                            common_s = set(lst_day_c) & set(lst_day)
                            for i in common_s:
                                lst_day_c.remove(i)
                            dem_khong = len(lst_day_c)
                            if dem <= len(set(lst_day)) and dem_khong == 0:
                                lst_df1.append(row)

                    else:
                        lst_df1.append(row)

    df_lst_df1 = pd.DataFrame(lst_df1)
    return df_lst_df1

# 14. Find filter name


def typeFilter_Name(typeFilter):
    if typeFilter == 'top':
        return ""
    return "Learning Path consists of "
