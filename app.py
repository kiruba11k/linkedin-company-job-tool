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
from concurrent.futures import ThreadPoolExecutor


# -------------------------
# SCRAPER FUNCTIONS
# -------------------------

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

            description_tag = (
                soup.select_one(".description__text")
                or soup.select_one(".show-more-less-html__markup")
            )

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

        link_el = job_li_element.select_one(
            'a[data-tracking-control-name="public_jobs_jserp-result_search-card"]'
        )
        title_el = job_li_element.select_one("h3.base-search-card__title")
        company_el = job_li_element.select_one("h4.base-search-card__subtitle")
        location_el = job_li_element.select_one("span.job-search-card__location")
        date_el = job_li_element.select_one("time.job-search-card__listdate")

        job_url = link_el["href"].split('?')[0] if link_el else None

        job_postings.append({
            "title": title_el.text.strip() if title_el else None,
            "company": company_el.text.strip() if company_el else None,
            "location": location_el.text.strip() if location_el else None,
            "publication_date": date_el["datetime"] if date_el else None,
            "url": job_url,
            "description": "Pending"
        })

    return job_postings


# -------------------------
# PARALLEL DESCRIPTION SCRAPING
# -------------------------

def fetch_descriptions_parallel(jobs, status_box):

    def task(job):
        if job["url"]:
            status_box.update(
                label=f"Fetching description: {job['title']}",
                state="running"
            )
            job["description"] = fetch_job_description(job["url"])
            time.sleep(1)
        else:
            job["description"] = "No URL found"

        return job

    with ThreadPoolExecutor(max_workers=10) as executor:
        jobs = list(executor.map(task, jobs))

    return jobs


# -------------------------
# ERP PROMPT
# -------------------------

ERP_PROMPT = """
You are an expert ERP job filter.

Task:
Analyze each job description and determine if it is a TRUE ERP-SPECIFIC ROLE.

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

Return STRICT JSON array only in this format:

[
  {
    "erp_match": "YES",
    "reason": "short explanation"
  }
]
"""


# -------------------------
# BATCH LLM FILTER
# -------------------------

def batch_llm_filter(jobs, client):

    try:

        batched_input = []

        for idx, job in enumerate(jobs):
            batched_input.append(
                f"""
                JOB {idx + 1}
                Title: {job['title']}
                Description: {job['description']}
                """
            )

        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": ERP_PROMPT
                },
                {
                    "role": "user",
                    "content": "\n\n".join(batched_input)
                }
            ]
        )

        result = completion.choices[0].message.content.strip()

        parsed = json.loads(result)

        if isinstance(parsed, dict):
            parsed = parsed.get("results", [])

        return parsed

    except Exception as e:
        print("Batch LLM error:", e)

        return [
            {
                "erp_match": "NO",
                "reason": "LLM parsing error"
            }
            for _ in jobs
        ]


# -------------------------
# MAIN SCRAPER PIPELINE
# -------------------------

def run_scraper(keyword, location, pages, api_key, status_box, progress_bar):

    client = Groq(api_key=api_key)

    all_jobs = []

    total_steps = pages
    step = 0

    status_box.update(label="Initializing scraping pipeline", state="running")

    for i in range(pages):

        start = i * 25

        try:

            status_box.update(
                label=f"Fetching LinkedIn page {i + 1}",
                state="running"
            )

            html = fetch_linkedin_jobs(keyword, location, start)

            status_box.update(
                label="Parsing job postings",
                state="running"
            )

            jobs = parse_job_postings(html)

            if not jobs:
                break

            status_box.update(
                label="Scraping job descriptions in parallel",
                state="running"
            )

            jobs = fetch_descriptions_parallel(jobs, status_box)

            status_box.update(
                label="Running batch LLM filtering",
                state="running"
            )

            batch_size = 10

            for batch_start in range(0, len(jobs), batch_size):

                batch_jobs = jobs[batch_start: batch_start + batch_size]

                llm_results = batch_llm_filter(batch_jobs, client)

                for job, llm_result in zip(batch_jobs, llm_results):

                    job["erp_match"] = llm_result.get("erp_match", "NO")
                    job["reason"] = llm_result.get("reason", "No reason")

                    if job["erp_match"] == "YES":
                        all_jobs.append(job)

            step += 1
            progress_bar.progress(step / total_steps)

            time.sleep(1)

        except Exception as e:
            print(e)

    status_box.update(label="Preparing final dataset", state="running")

    df = pd.DataFrame(all_jobs)

    df.to_csv("final_erp_jobs.csv", index=False)

    st.session_state["data"] = df

    progress_bar.progress(1.0)

    status_box.update(label="Scraping completed", state="complete")


# -------------------------
# STREAMLIT UI
# -------------------------

st.set_page_config(page_title="ERP Hiring Intelligence Scraper", layout="wide")

st.title("ERP Hiring Intelligence Scraper")

st.write("Scrape LinkedIn ERP hiring signals and filter using Groq LLM")

keyword = st.text_input(
    "Keyword",
    "sap hiring in MAG (Airports Group)"
)

location = st.text_input(
    "Location",
    "United Kingdom"
)

pages = st.number_input(
    "Pages to Scrape",
    min_value=1,
    max_value=10,
    value=2
)

groq_key = st.secrets.get("Groq")


# -------------------------
# START SCRAPER
# -------------------------

if st.button("Start Scraping"):

    st.session_state["data"] = None

    status_box = st.status("Starting pipeline", expanded=True)

    progress_bar = st.progress(0)

    thread = threading.Thread(
        target=run_scraper,
        args=(
            keyword,
            location,
            pages,
            groq_key,
            status_box,
            progress_bar
        )
    )

    thread.start()

    st.write("Scraping is running in background.")


# -------------------------
# OUTPUT RESULTS
# -------------------------

if "data" in st.session_state and st.session_state["data"] is not None:

    df = st.session_state["data"]

    st.subheader("Filtered ERP Jobs")

    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="erp_jobs_filtered.csv",
        mime="text/csv"
    )
