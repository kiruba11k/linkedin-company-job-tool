import streamlit as st
import pandas as pd
import threading
import requests
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
    except:
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
        print(e)

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

    return response.text


def parse_job_postings(html):

    soup = BeautifulSoup(html, "html.parser")

    jobs = []

    for job_li in soup.select("li"):

        link_el = job_li.select_one(
            'a[data-tracking-control-name="public_jobs_jserp-result_search-card"]'
        )

        title_el = job_li.select_one("h3.base-search-card__title")
        company_el = job_li.select_one("h4.base-search-card__subtitle")
        location_el = job_li.select_one("span.job-search-card__location")
        date_el = job_li.select_one("time.job-search-card__listdate")

        job_url = link_el["href"].split('?')[0] if link_el else None

        jobs.append({
            "title": title_el.text.strip() if title_el else None,
            "company": company_el.text.strip() if company_el else None,
            "location": location_el.text.strip() if location_el else None,
            "publication_date": date_el["datetime"] if date_el else None,
            "url": job_url,
            "description": ""
        })

    return jobs


# -------------------------
# PARALLEL DESCRIPTION SCRAPE
# -------------------------

def fetch_descriptions_parallel(jobs):

    def task(job):
        job["description"] = fetch_job_description(job["url"])
        return job

    with ThreadPoolExecutor(max_workers=10) as executor:
        jobs = list(executor.map(task, jobs))

    return jobs


# -------------------------
# ERP PROMPT
# -------------------------

ERP_PROMPT = """
You are an expert ERP job filter.

Analyze the job description and determine if it is a TRUE ERP-SPECIFIC ROLE.

Return JSON:

{
"erp_match": "YES or NO",
"reason": "short explanation"
}
"""


# -------------------------
# LLM ANALYSIS
# -------------------------

def analyze_jobs_llm(df, client, status_box):

    results = []

    for i, row in df.iterrows():

        status_box.update(
            label=f"Analyzing job {i+1}/{len(df)} using LLM",
            state="running"
        )

        try:

            completion = client.chat.completions.create(

                model="llama3-70b-8192",

                response_format={"type": "json_object"},

                messages=[
                    {"role": "system", "content": ERP_PROMPT},
                    {"role": "user", "content": row["description"]}
                ]
            )

            res = json.loads(completion.choices[0].message.content)

            results.append(res)

        except:

            results.append({
                "erp_match": "NO",
                "reason": "LLM error"
            })

    df["erp_match"] = [r["erp_match"] for r in results]
    df["reason"] = [r["reason"] for r in results]

    return df


# -------------------------
# MAIN PIPELINE
# -------------------------

def run_scraper(keyword, location, pages, api_key, status_box, progress_bar):

    client = Groq(api_key=api_key)

    all_jobs = []

    status_box.update(label="Scraping LinkedIn jobs", state="running")

    for page in range(pages):

        start = page * 25

        html = fetch_linkedin_jobs(keyword, location, start)

        jobs = parse_job_postings(html)

        jobs = fetch_descriptions_parallel(jobs)

        all_jobs.extend(jobs)

        progress_bar.progress((page+1)/pages)

    # -------------------------
    # SHOW SCRAPED DATA FIRST
    # -------------------------

    scraped_df = pd.DataFrame(all_jobs)

    st.session_state["scraped"] = scraped_df

    status_box.update(label="Scraping completed. Starting LLM analysis", state="running")

    # -------------------------
    # LLM ANALYSIS
    # -------------------------

    final_df = analyze_jobs_llm(scraped_df, client, status_box)

    final_df.to_csv("final_erp_jobs.csv", index=False)

    st.session_state["data"] = final_df

    status_box.update(label="LLM analysis completed", state="complete")


# -------------------------
# STREAMLIT UI
# -------------------------

st.set_page_config(page_title="ERP Hiring Intelligence Scraper", layout="wide")

st.title("ERP Hiring Intelligence Scraper")

keyword = st.text_input("Keyword", "sap hiring in MAG (Airports Group)")

location = st.text_input("Location", "United Kingdom")

pages = st.number_input("Pages", 1, 10, 2)

groq_key = st.secrets.get("Groq")


# -------------------------
# START SCRAPER
# -------------------------

if st.button("Start Scraping"):

    status_box = st.status("Starting pipeline", expanded=True)

    progress_bar = st.progress(0)

    thread = threading.Thread(
        target=run_scraper,
        args=(keyword, location, pages, groq_key, status_box, progress_bar)
    )

    thread.start()


# -------------------------
# SHOW SCRAPED RESULTS
# -------------------------

if "scraped" in st.session_state:

    st.subheader("Scraped Jobs (Before LLM Analysis)")

    st.dataframe(st.session_state["scraped"], use_container_width=True)


# -------------------------
# SHOW FINAL RESULTS
# -------------------------

if "data" in st.session_state:

    st.subheader("Final Results With LLM Analysis")

    df = st.session_state["data"]

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        csv,
        "erp_jobs_filtered.csv",
        "text/csv"
    )
