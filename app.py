
from flask import Flask, render_template, request, jsonify
import openai
import json
import pandas as pd
import os

app = Flask(__name__)

# Get the API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the API key
openai.api_key = openai_api_key

#Defining Temperature and Seed

seed = 42

temperature_extraction_skills = .5
temperature_extraction_verbs = .3
temperature_compare = .3
temperature_rewrite = .7
best_of_var = 5


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-feature1', methods=['POST'])
def process_feature1():
    resume = request.form['resume']
    job_description = request.form['job_description']

    resume_skills_tools, resume_action_verbs = extract_skills_verbs(resume)
    jd_skills_tools, jd_action_verbs = extract_skills_verbs(job_description)

    # Use OpenAI API to compare skills and verbs
    skills_comparison_text = compare_skills(resume_skills_tools, jd_skills_tools)
    verbs_comparison_text = compare_verbs(resume_action_verbs, jd_action_verbs)

    # # Parse the text to DataFrames
    # skills_comparison_df = parse_table_to_df(skills_comparison_text)
    # #print('skills df cols',skills_comparison_df.columns)
    # #print('skills df', skills_comparison_df)
    # verbs_comparison_df = parse_table_to_df(verbs_comparison_text)
    # #print('verbs df cols', verbs_comparison_df.columns)
    # #print('verbs df', verbs_comparison_df)

    # Convert DataFrames to lists of dictionaries for template rendering
    skills_comparison = json.loads(skills_comparison_text)
    verbs_comparison = json.loads(verbs_comparison_text)
    print('3. Skill Comparison Being Rendered ', skills_comparison)
    #print('4. Verb Comparison Being Rendered ', verbs_comparison)
    # Render and return the HTML snippet with the data
    return render_template('feature1_results_snippet.html',
                           resume_skills=resume_skills_tools,
                           jd_skills=jd_skills_tools,
                           resume_verbs=resume_action_verbs,
                           jd_verbs=jd_action_verbs,
                           skills_comparison=skills_comparison,
                           verbs_comparison=verbs_comparison)

@app.route('/process-feature2', methods=['POST'])
def process_feature2():
    resume_point = request.form['resume_point']
    skill = request.form['skill']
    action_verb = request.form['action_verb']
    rewritten_point = rewrite_resume_point(resume_point, skill, action_verb)

    # Render and return the HTML snippet with the rewritten point
    return render_template('feature2_results_snippet.html', rewritten_point=rewritten_point)

def extract_skills_verbs(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":  """You are a LinkedIN consultant. 
                                    Input: Candidate Resume
                                    Output: a. Key skills/tools 
                                             b. Only important action verbs.
                                    Output conditions:
                                    1. Be concise (maximum 3 words per bullet)
                                    2. Minimum 7 bullets
                                    3. Maximum 10 bullets
                                    4. Combine duplicates e.g. AI and AI Modelling are same skill
                                    5. Ignore: Minor verbs like Need, Working, Helping etc.
                                    6. Ignore: Abilities that you would expect an MBA student to have, for example: English fluency
                                    7. Ignore: Job titles 
                                    8. Mention technical skills in same bullet e.g. R, Python, etc. 
                                    9. Sort 2 lists by order of relative importance of skill or action verb in document
                                    10. Number the lists
                                    11. Skill could include area of expertise (e.g. Climate Technology), """},
            {"role": "user", "content": text}
        ],
        temperature = temperature_extraction_skills,
        seed = seed
    )
    content = response.choices[0].message.content
    sections = content.split('\n\n')
    skills_tools = [skill.strip('1234567890. ') for skill in sections[0].split('\n')[1:]]
    action_verbs = [verb.strip('1234567890. ') for verb in sections[1].split('\n')[1:]]
    return skills_tools, action_verbs

def compare_skills(resume_skills, jd_skills):
    # Convert lists to JSON for OpenAI API input
    resume_skills_json = json.dumps(resume_skills, ensure_ascii=False)
    jd_skills_json = json.dumps(jd_skills, ensure_ascii=False)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """""You are a LinkedIN consultant.
                                     Input: 
                                     Table 1: Skills/tools required according to Job description
                                     Table 2: Skills/tools demonstrated in resume
                                     Output:
                                     Format: Table
                                     Column1: Skills/tools required according to Job description
                                     Column2: Skills/tools demonstrated in resume that match skills/tools required according to job description
                                     Column3: Strength of matching: High/Medium/Low
                                     
                                     Column Names:
                                     1. JD Skills
                                     2. Resume Skills
                                     3. Strength
                                     Output conditions: 
                                     1. Be generous in matching. Eg. If table2 includes R, Python, impact measurement, they will know maths and statistics 
                                     2. Sort by descending in Column 3 
                                     3. Include all JD Skills in Column 1
                                     
                                     """
            },
            {
                "role": "user",
                "content": f"Table1: {jd_skills_json}\n\nTable2: {resume_skills_json}"
            }
        ],
        temperature=temperature_extraction_verbs,
        seed=seed,
        n = 3
    )
    #print(response.choices[0].message.content)
    response_text = response.choices[0].message.content
    return convert_response_to_json(response_text)


def compare_verbs(resume_verbs, jd_verbs):
    # Convert lists to JSON for OpenAI API input
    resume_verbs_json = json.dumps(resume_verbs, ensure_ascii=False)
    jd_verbs_json = json.dumps(jd_verbs, ensure_ascii=False)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """""You are a LinkedIN consultant.
                                     Input: 
                                     Table 1: Action Verbs used in Job description
                                     Table 2: Action Verbs used in resume
                                     Output:
                                     Format: Table
                                     Column1: Action Verbs used in Job description
                                     Column2: Action Verbs used in resume that match Action Verbs used in job description
                                     Column3: Strength of matching: High/Medium/Low
                                     
                                     
                                     Column Names:
                                     1. JD Verbs
                                     2. Resume Verbs
                                     3. Strength

                                     Output conditions: 
                                     1. Be generous in matching. Eg. If table2 includes R, Python, impact measurement, they will know maths and statistics 
                                     2. Sort by descending in Column 3
                                     3. Include all JD Verbs in Column 1

                                     """
            },
            {
                "role": "user",
                "content": f"Table1: {jd_verbs_json}\n\nTable2: {resume_verbs_json}"
            }
        ],
        temperature=temperature_compare,
        seed=seed
    )
    print('1. Raw Comparison Output',response.choices[0].message.content)
    response_text = response.choices[0].message.content
    print('2. JSON of Comparison Output',convert_response_to_json(response_text))
    return convert_response_to_json(response_text)

def rewrite_resume_point(point, skill, action_verb):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Rewrite the resume point considering the given skill and action verb."},
            {"role": "user", "content": f"1. Resume Point: {point}\n2. Skill: {skill}\n3. Action Verb: {action_verb}"}
        ],
        temperature=temperature_rewrite,
        seed=seed
    )
    #print(response.choices[0].message.content)
    return response.choices[0].message.content

def parse_table_to_df(table_text):
    lines = table_text.strip().split('\n')
    headers = [header.strip() for header in lines[0].split('|') if header.strip()]
    rows = []

    for line in lines[2:]:
        row_data = [data.strip() for data in line.split('|') if data.strip()]
        if row_data:
            rows.append(row_data)

    return pd.DataFrame(rows, columns=headers)

def convert_response_to_json(response_text):
    df = parse_table_to_df(response_text)
    #print(df.to_json(orient='records'))
    return df.to_json(orient='records')

if __name__ == '__main__':
    # Use os.environ.get() to get the port dynamically
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug = True)
