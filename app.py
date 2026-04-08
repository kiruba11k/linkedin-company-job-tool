import streamlit as st
import pandas as pd
import threading
import requests
import csv
import time
import json
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from groq import Groq




def can_fetch(url, user_agent='*'):
    rp = RobotFileParser()
    try:
        rp.set_url("https://www.linkedin.com/robots.txt")
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return False


def fetch_job_description(job_url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(job_url, headers=headers, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            description_tag = soup.select_one(".description__text") or soup.select_one(".show-more-less-html__markup")

            return description_tag.get_text(separator="\n", strip=True) if description_tag else "Description not found"

    except Exception as e:
        print(f"Could not fetch description for {job_url}: {e}")

    return "Error fetching description"


def fetch_linkedin_jobs(keyword, location, start=0):
    base_url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"

    params = {
        "keywords": keyword,
        "location": location,
        "start": start
    }

    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(base_url, params=params, headers=headers)
    response.raise_for_status()

    return response.text


def parse_job_postings(job_postings_html):

    job_postings = []

    soup = BeautifulSoup(job_postings_html, "html.parser")

    job_li_elements = soup.select("li")

    for job_li_element in job_li_elements:

        link_el = job_li_element.select_one('a[data-tracking-control-name="public_jobs_jserp-result_search-card"]')
        title_el = job_li_element.select_one("h3.base-search-card__title")
        company_el = job_li_element.select_one("h4.base-search-card__subtitle")
        location_el = job_li_element.select_one("span.job-search-card__location")
        date_el = job_li_element.select_one("time.job-search-card__listdate")

        job_url = link_el["href"].split('?')[0] if link_el else None

        description = "N/A"

        if job_url:
            description = fetch_job_description(job_url)
            time.sleep(10)

        job_postings.append({
            "title": title_el.text.strip() if title_el else None,
            "company": company_el.text.strip() if company_el else None,
            "location": location_el.text.strip() if location_el else None,
            "publication_date": date_el["datetime"] if date_el else None,
            "url": job_url,
            "description": description
        })

    return job_postings


def save_to_csv(data, filename="linkedin_jobs.csv"):
    if not data:
        return

    keys = data[0].keys()

    with open(filename, 'w', newline='', encoding='utf-8') as output_file:

        dict_writer = csv.DictWriter(output_file, fieldnames=keys)

        dict_writer.writeheader()

        dict_writer.writerows(data)





ERP_PROMPT = """
You are an expert ERP job filter.

Task: Analyze the job description and determine if it is a TRUE ERP-SPECIFIC ROLE.

STRICT CRITERIA:

Only say YES if:
- ERP is the CORE responsibility
- Includes ERP implementation, rollout, migration
- ERP configuration or customization
- ERP functional consulting
- ERP architecture or ownership
- ERP transformation program
- ERP integration
- ERP module ownership

Say NO if:
- ERP is just a tool
- Finance / accounting roles
- PMO or operations
- Testing only
- Master data roles
- ERP mentioned as experience

Return JSON:

{
"erp_match": "YES or NO",
"reason": "short explanation"
}
"""


# -------------------------
# GROQ ERP FILTER
# -------------------------

def erp_filter(description, client):

    try:

        completion = client.chat.completions.create(

            model="Llama 3.3 70B Versatile",

            response_format={"type": "json_object"},

            messages=[

                {"role": "system", "content": ERP_PROMPT},

                {"role": "user", "content": description}

            ]
        )

        result = completion.choices[0].message.content

        data = json.loads(result)

        return data["erp_match"], data["reason"]

    except Exception as e:

        return "NO", "LLM parsing error"



# -------------------------
# MAIN SCRAPER PIPELINE
# -------------------------

def run_scraper(company, location, pages, api_key):

    client = Groq(api_key=api_key)

    erp_keywords = ["sap", "workday", "oracle", "dynamics", "infor", "sage"]

    all_jobs = []

    for erp in erp_keywords:

        keyword = f"{erp} hiring in {company}"

        for i in range(pages):

            start = i * 25

            try:

                html = fetch_linkedin_jobs(keyword, location, start)

                jobs = parse_job_postings(html)

                if not jobs:
                    break

                for job in jobs:

                    decision, reason = erp_filter(job["description"], client)

                    job["erp_match"] = decision
                    job["reason"] = reason

                    if decision == "YES":
                        all_jobs.append(job)

                time.sleep(5)

            except Exception as e:
                print(e)

    df = pd.DataFrame(all_jobs)

    df.to_csv("final_erp_jobs.csv", index=False)

    st.session_state["data"] = df



# -------------------------
# STREAMLIT UI
# -------------------------

st.title("ERP Hiring Intelligence Scraper")

st.write("Scrape LinkedIn ERP hiring signals and filter using Groq LLM")

company = st.text_input("Company Name", "MAG (Airports Group)")

location = st.text_input("Location", "United Kingdom")

pages = st.number_input("Pages to Scrape", 1, 10, 2)

groq_key = st.secrets.get("Groq")


if st.button("Start Scraping"):

    st.session_state["data"] = None

    thread = threading.Thread(
        target=run_scraper,
        args=(company, location, pages, groq_key)
    )

    thread.start()

    st.success("Scraping started in background. You can switch tabs.")



# -------------------------
# OUTPUT TABLE
# -------------------------

if "data" in st.session_state and st.session_state["data"] is not None:

    df = st.session_state["data"]

    st.subheader("Filtered ERP Jobs")

    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="Download CSV",

        data=csv,

        file_name="erp_jobs_filtered.csv",

        mime="text/csv"
    )
