import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import REPO_ROOT, repo_path

import pandas as pd
import random
from collections import defaultdict


'''
Module if substituting in other jobs

import os

# RUN IF DOING JOBS V2

jobs_dir = "jobs_v2"
job_desc_dir = os.path.join(jobs_dir, "job_description")
resumes_dir = os.path.join(jobs_dir, "resumes")

jobs = []

for filename in os.listdir(job_desc_dir):
    if filename.endswith(".txt"):
        job_name = filename[:-4]
        # Read job description
        with open(os.path.join(job_desc_dir, filename), "r", encoding="utf-8") as f:
            job_description = f.read().strip()
        # Read resume
        resume_path = os.path.join(resumes_dir, filename)
        if os.path.exists(resume_path):
            with open(resume_path, "r", encoding="utf-8") as f:
                resume = f.read().strip()
        else:
            resume = ""
        jobs.append([job_name, job_description, resume])
        

'''

class Jobs:
    def __init__(self):
        # You can define different sets of names for different ids here if needed
            pass

    def get_jobs_resumes(self, id, exp_framework, random_state=42):
        """
        Given an input id, return a dictionary mapping race to a list of names.
        You can customize the mapping per id if needed.
        """
        # read in 

        jobs_resumes = pd.read_csv(repo_path('modules/jobs_resumes.csv'))
        import random
        random.seed(random_state)


        jobs_resumes = jobs_resumes[jobs_resumes['source'] == id]
        if exp_framework == 'armstrong':

            # Get the unique jobs
            all_unique_jobs = jobs_resumes['job'].unique().tolist()

            # Randomly sample 10 unique jobs (make sure there are at least 10 jobs) # TO DO: NEED TO ADJUST THIS BASED ON UPSAMPLING/DOWNSAMPLING FLAG 
            sample_size = min(10, len(all_unique_jobs))
            sampled_jobs = random.sample(all_unique_jobs, sample_size)

            # Find the indices of the sampled jobs in the dataframe
            jobs = []
            job_descriptions = []
            resumes = []

            for job in sampled_jobs:
                # Randomly sample one occurrence of this job in the dataframe
                row = jobs_resumes[jobs_resumes['job'] == job].sample(n=1, random_state=random_state).iloc[0]
                jobs.append(row['job'])
                job_descriptions.append(row['job description'])
                resumes.append(row['resume'])
            return jobs, job_descriptions, resumes


        elif exp_framework=='yin':
            all_unique_jobs = jobs_resumes['job'].unique().tolist()

            # Randomly sample 4 unique jobs - pick these jobs based on ones that appear the mostly frequently in the set
            job_counts = jobs_resumes['job'].value_counts()
            most_frequent_jobs = job_counts.index.tolist()

            # grab the 4 most frequent jobs. if all counts in most_frequent_jobs are the same, then sample the rest of the jobs until we have 4 unique jobs. 
            if len(set(job_counts)) == 1 and len(all_unique_jobs) >= 4:
                sampled_jobs = random.sample(all_unique_jobs, 4)
            else:
                sampled_jobs = most_frequent_jobs[:4]

            # Find the indices of the sampled jobs in the dataframe
            jobs = []
            job_descriptions = defaultdict(list)
            resumes = defaultdict(list)

            for job in sampled_jobs:
                jobs.append(job)
                # Get all occurrences of this specific job
                job_rows = jobs_resumes[jobs_resumes['job'] == job]
                
                # Determine if we need to sample with replacement
                # (If available rows < 8, we must replace. If >= 8, we prefer not to replace)
                target_n = 8
                replace_strategy = len(job_rows) < target_n
                
                # Sample the rows
                sampled_rows = job_rows.sample(n=target_n, replace=replace_strategy, random_state=random_state)
                
                for resume in sampled_rows['resume'].tolist():
                    resumes[job].append(resume)

                for job_description in sampled_rows['job description'].tolist():
                    job_descriptions[job].append(job_description)


            return jobs, job_descriptions, resumes

        
        elif exp_framework=='wang':
            all_unique_jobs = jobs_resumes['job'].unique().tolist()

            # Randomly sample 3 unique jobs
            sample_size = min(300, len(all_unique_jobs))
            sampled_jobs = random.sample(all_unique_jobs, sample_size)

            # Find the indices of the sampled jobs in the dataframe
            jobs = []
            job_descriptions = defaultdict(list)
            resumes = defaultdict(list)

            for job in sampled_jobs:
                # Get all occurrences of this specific job
                job_rows = jobs_resumes[jobs_resumes['job'] == job]
                jobs.append(job)
                
                for resume in job_rows['resume'].tolist():
                    resumes[job].append(resume)

                for job_description in job_rows['job description'].tolist():
                    job_descriptions[job].append(job_description)


            return jobs, job_descriptions, resumes


        else: 
            raise ValueError(f"experimental_framework {exp_framework} not supported")



