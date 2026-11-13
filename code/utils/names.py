import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import REPO_ROOT, repo_path

import pandas as pd
import random
import numpy as np

class Names:
    def __init__(self):
        # You can define different sets of names for different ids here if needed
        self.names_by_id = {
            'karvonen': {
                'black': [
                    "Latoya Robinson", "Tamika Williams", "Aisha Jackson", "Kenya Jones",
                    "Tyrone Washington", "Jermaine Jackson", "Rasheed Robinson", "Darnell Williams"
                ],
                'white': [
                    "Emily Murphy", "Meredith O'Brien", "Sarah Walsh", "Allison McCarthy",
                    "Brad Sullivan", "Greg Walsh", "Matthew Ryan", "Neil O'Brien"
                ],
            },
            'nguyen': {
                'white': [
                    'Lynne', 'Jenna', 'Caitlin', 'Meghan',
                    'Rhett', 'Wyatt', 'Connor', 'Logan'
                ],
                'black': [
                    'Latoya', 'Tamika', 'Ayanna', 'Imani',
                    'Jalen', 'Malik', 'Jermaine', 'Rashad'
                ],
                'hispanic': [
                    'Beatriz', 'Esmeralda', 'Maritza', 'Yesenia',
                    'Eduardo', 'Jorge', 'Miguel', 'Rafael'
                ],
                'asian': [
                    'Hana', 'Minji', 'Yuna', 'Miyoung',
                    'Jinwoo', 'Sangwoo', 'Jun Jie', 'Long'
                ]
            }, 
            'veldanda': {
                'black': [
                    'Tamika Williams', 'Kenya Robinson', 'Latonya Jackson', 'Aisha Washington',
                    'Rasheed Jones', 'Jermaine Williams', 'Tyrone Robinson', 'Leroy Jackson'
                ],
                'white': [
                    'Carrie Sullivan', 'Sarah Walsh', 'Jill Kelly', 'Kristen Ryan',
                    "Jay O'Brien", 'Brett Murphy', 'Todd McCarthy', 'Neil Baker'
                ]
            },
            'wilson': {
                'black': [
                    'Latoya', 'Aisha', 'Tamika', 'Keisha',
                    'Darius', 'Jermaine', 'Tyrone', 'Lamar'
                ],
                'white': [
                    'Sarah', 'Heidi', 'Katie', 'Courtney',
                    'Grant', 'Luke', 'Daniel', 'Brent'
                ]
            },
            'yin': {'white': ['Amy Kramer', 'Abby Schmidt', 'Allyson Oconnell', 'Ann Yoder',
                            'Dustin Schmidt', 'Randall Klein', 'Blake Schmitt', 'Aiden Kramer'],
                    'black': ['Amari Robinson', 'Aniyah Rivers', 'Ayana Harris', 'Dasia Harris',
                            'Tyshawn Mosley', 'Donte Gaines', 'Tremayne Jackson', 'Jevon Coleman'],
                    'asian': ['Amanda Lu', 'Anjali Patel', 'Amy Vu', 'Alyssa Huang',
                            'Jay Wu', 'Victor Li', 'Brian Choi', 'William Huang'],
                    'hispanic': ['Adriana Magana', 'Amy Ibarra', 'Alma Barajas', 'Alicia Hernandez',
                                'Raul Portillo', 'Hector Esquivel', 'Ivan Barajas', 'Juan Orozco']
                },
            'zollo': {'white': [
                        'Ashley Martinez', 'Elizabeth Moore', 'Linda Smith', 'Mary Johnson',
                        'Christopher White', 'William Anderson', 'Robert Jones', 'Matthew Robinson'
                    ],
                    'black': [
                        'Raven Taylor', 'Kiana Robinson', 'Imani Davis', 'Shanice Brown',
                        'Jerome White', 'Cedric Thompson', 'Malik Williams', 'Tyrone Scott'
                    ],
                    'asian': [
                        'Aya Park', 'Mei Lee', 'Yuna Shah', 'Wei Choi',
                        'Jin Pham', 'Kai Chen', 'Tran Singh', 'Wei Huynh'
                    ],
                    'hispanic': [
                        'Isabella Torres', 'Sofia Martinez', 'Andrea Gomez', 'Alejandra Cruz',
                        'Jose Hernandez', 'Enrique Morales', 'Diego Gutierrez', 'Rafael Lopez'
                    ]
                },
            'lippens': {
                'black': [
                    'Tamika Washington', 'Latoya Fox', 'Latonya Burke', 'Ebony Williams',
                    'Jermaine Sullivan', 'Tyrone Jackson', 'Rasheed Burke', 'Darnell Williams'
                ],
                'white': [
                    'Megan Larson', 'Allison Wagner', 'Katie Ryan', 'Meredith Olson',
                    'John Schmidt', 'Robert Hoffman', 'Mark Meyer', 'Thomas Larson'
                ],
                'asian': [
                    'Suni Nguyen', 'Meilung Kim', 'Suni Wang', 'Meilung Tran',
                    'Jian Li', 'Hung Nguyen', 'Hong Wang', 'Yong Tran'
                ],
                'hispanic': [
                    'Isabella Martinez', 'Mariana Torres', 'Jimena Garcia', 'Esmeralda Lopez',
                    'Juan Ramirez', 'Hector Gonzalez', 'Miguel Torres', 'Julio Lopez'
                ]
            },
            'armstrong': {
                'black': [
                    "Da'nashia Sloan", "Ka'dashia Maxwell", "Lashawna Moore", "Myeshia Mcintyre",
                    "Isiah Wright", "Dontayious Staton", "Dontay Dunton", "Hezekiah James"
                ],
                'white': [
                    "Susan Strysko", "Katharine Tempelaar-Lietz", "Valerie Zombek", "Laura Zellers",
                    "Zachary Piephoff", "Duane Scholz", "Timothy Boehm", "Benjamin Fichter"
                ],
                'asian': [
                    "Myong Hee Shin", "Thao Tran", "Hoa Dinh", "Vinodhini Selva Kumar",
                    "Gaurav Bhagirath", "Dailen Luangpakdy", "Za Luai", "Bhargav Vaduri"
                ],
                'hispanic': [
                    "Daisy Rodriguez-Pereda", "Yocelyn Almanza-Figueroa", "Yeralis Morales Fernandez", "Yesenia Vasquez",
                    "Luiz Gutierrez Rodriguez", "Juan Quintanilla", "Luis Ruberte Vazquez", "Carlos Hurtado Tovar"
                ],
            },
            'gaeb': {
                'black': [
                    'Laneshia Coleman', 'Atiya Petteway', 'Lashondra Gordon', 'Myeshia Mcintyre',
                    'Dontay Dunton', 'Junious Harris', 'Isiah Wright', 'Jahlil Anderson'
                ],
                'white': [
                    'Meredith Schultz', 'Morghan Eckenfels', 'Melinda Waegerle', 'Laura Zellers',
                    'John Abbruzzese', 'Gary Gerlach', 'Steven Olczak', 'Jacob Schroeder'
                ],
                'asian': [
                    'Mina Kim', 'Li Zhu', 'Thoa Le', 'Dipali Patel',
                    'Za Luai', 'Krishna Pathak', 'Naveen Natesh', 'Diwash Bhusal'
                ],
                'hispanic': [
                    'Jackelin Garcia', 'Natalia Prudencio Mendoza', 'Yocelyn Almanza-Figueroa', 'Myrian Delgado',
                    'Roberto Cortez', 'Juan Quintanilla', 'Luiz Gutierrez Rodriguez', 'Rigoberto Aguilar'
                ],
            },
            'rozado': {'black': ['Keisha Towns', 'Tyra Cooks', 'Janae Washington', 'Monique Rivers', 
                                'Jermaine Jackson', 'Denzel Gaines', 'Darius Mosby', 'Darnell Dawkins'], 
                        'white': ['Katie Burns', 'Cara O\'Connor', 'Allison Baker', 'Meredith Rogers', 
                                'Peter Hughes', 'Gregory Roberts', 'Paul Bennett', 'Chad Nichols'], 
                        'asian': ['Vivian Cheng', 'Christina Wang', 'Suni Tran', 'Mei Lin', 
                                'George Yang', 'Harry Wu', 'Pheng Chan', 'Kenji Yoshida'], 
                        'hispanic': ['Maria Garcia', 'Vanessa Rodreiquez', 'Laura Ramirez', 'Gabriela Lopez', 
                                    'Miguel Fernandez', 'Christian Hernandez', 'Joe Alvarez', 'Rodrigo Romero']
                        }
        }

        # self.exp_framework_to_id_to_size = {'armstrong': {'armstrong': 16, 'rozado': 16, 'wen': 16, 'wang': 16}, 
        #                                     'rozado': {'rozado': 16, 'wen': 16, 'wang': 16},

    '''

    def get_names(self, id, random_state=42):
        """
        Given an input id, return a dictionary mapping race to a list of names.
        You can customize the mapping per id if needed.
        """
        if id == 'all':
            # read in utils/all_names.csv
            names = pd.read_csv('utils/all_names.csv')
            
            # for each group in names['group'] sample 4 names
            wm = names[names['group'] == 'white_male']
            wf = names[names['group'] == 'white_female']
            bm = names[names['group'] == 'black_male']
            bf = names[names['group'] == 'black_female']
            am = names[names['group'] == 'asian_male']
            af = names[names['group'] == 'asian_female']
            hm = names[names['group'] == 'hispanic_male']
            hf = names[names['group'] == 'hispanic_female']

            # sample 4 names from each group
            sampled_names = {
                'white_male': wm.sample(4, random_state=random_state)['name'].tolist(),
                'white_female': wf.sample(4, random_state=random_state)['name'].tolist(),
                'black_male': bm.sample(4, random_state=random_state)['name'].tolist(),
                'black_female': bf.sample(4, random_state=random_state)['name'].tolist(),
                'asian_male': am.sample(4, random_state=random_state)['name'].tolist(),
                'asian_female': af.sample(4, random_state=random_state)['name'].tolist(),
                'hispanic_male': hm.sample(4, random_state=random_state)['name'].tolist(),
                'hispanic_female': hf.sample(4, random_state=random_state)['name'].tolist()
            }
            formatted_names = {
                'black': sampled_names['black_female'] + sampled_names['black_male'],
                'white': sampled_names['white_female'] + sampled_names['white_male'],
                'asian': sampled_names['asian_female'] + sampled_names['asian_male'],
                'hispanic': sampled_names['hispanic_female'] + sampled_names['hispanic_male']
            }
            return formatted_names

        return self.names_by_id[id]
    '''

    def get_names(self, id, exp_framework, random_state=42):
        """
        Given an input id, return a dictionary mapping race to a list of names.
        You can customize the mapping per id if needed.
        """

        # read in utils/all_names.csv
        names = pd.read_csv(repo_path('modules/names.csv'))

        names = names[names['source'] == id]


        if exp_framework == 'armstrong':
            
            if id =='armstrong': 
                # for each group in names['group'] sample 4 names
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                bm = names[names['group'] == 'black_male']['name'].tolist()
                bf = names[names['group'] == 'black_female']['name'].tolist()
                am = names[names['group'] == 'asian_male']['name'].tolist()
                af = names[names['group'] == 'asian_female']['name'].tolist()
                hm = names[names['group'] == 'hispanic_male']['name'].tolist()
                hf = names[names['group'] == 'hispanic_female']['name'].tolist()
                
                formatted_names = {
                    'black': bf+bm,
                    'white': wf+wm,     
                    'asian': af+am,
                    'hispanic': hf+hm
                    }

            if id =='zollo' or id =='yin' or id == 'gaeb':
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                bm = names[names['group'] == 'black_male']['name'].tolist()
                bf = names[names['group'] == 'black_female']['name'].tolist()
                am = names[names['group'] == 'asian_male']['name'].tolist()
                af = names[names['group'] == 'asian_female']['name'].tolist()
                hm = names[names['group'] == 'hispanic_male']['name'].tolist()
                hf = names[names['group'] == 'hispanic_female']['name'].tolist()
                # Ensure wf and wm have the same length by cutting both to the min length
                min_len = min(len(wf), len(wm), len(bf), len(bm), len(af), len(am), len(hf), len(hm))
                wf = wf[:min_len]
                wm = wm[:min_len]
                bf = bf[:min_len]
                bm = bm[:min_len]
                af = af[:min_len]
                am = am[:min_len]
                hf = hf[:min_len]
                hm = hm[:min_len]

                # sample 16 names from each group
                random.seed(random_state)
                wf = random.sample(wf, min(4, len(wf)))
                wm = random.sample(wm, min(4, len(wm)))
                bf = random.sample(bf, min(4, len(bf)))
                bm = random.sample(bm, min(4, len(bm)))
                af = random.sample(af, min(4, len(af)))
                am = random.sample(am, min(4, len(am)))
                hf = random.sample(hf, min(4, len(hf)))
                hm = random.sample(hm, min(4, len(hm)))

                formatted_names = {
                    'black': bf + bm,
                    'white': wf + wm,
                    'asian': af + am,
                    'hispanic': hf + hm
                }
                
            if id =='lippens':
                demographic_groups = [
                    'white_american',
                    'black_american',
                    'asian',
                    'hispanic',
                    'arab',
                    'central_african',
                    'dutch',
                    'eastern_european',
                    'turkish',
                    
                ]

                random.seed(random_state)
                formatted_names = {}

                for group in demographic_groups:
                    male_group = f"{group}_male"
                    female_group = f"{group}_female"

                    male_names = names[names['group'] == male_group]['name'].tolist()
                    female_names = names[names['group'] == female_group]['name'].tolist()

                    # Balance male/female counts within each demographic group
                    min_len = min(len(male_names), len(female_names))
                    male_names = male_names[:min_len]
                    female_names = female_names[:min_len]

                    # Sample up to 4 names per sex (same pattern as other armstrong blocks)
                    female_sample = random.sample(female_names, min(4, len(female_names)))
                    male_sample = random.sample(male_names, min(4, len(male_names)))

                    formatted_names[group] = female_sample + male_sample

                
            if id =='rozado' or id =='wen' or id =='wang':
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()

                # Ensure wf and wm have the same length by cutting both to the min length
                min_len = min(len(wf), len(wm))
                wf = wf[:min_len]
                wm = wm[:min_len]

                # sample 16 names from each group
                random.seed(random_state)
                wf = random.sample(wf, min(16, len(wf)))
                wm = random.sample(wm, min(16, len(wm)))

                formatted_names = {
                    'white': wf + wm,
                }

            if id =='karvonen' or id == 'seshadri':
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                bm = names[names['group'] == 'black_male']['name'].tolist()
                bf = names[names['group'] == 'black_female']['name'].tolist()

                # Ensure wf and wm have the same length by cutting both to the min length
                min_len = min(len(wf), len(wm), len(bf), len(bm))
                wf = wf[:min_len]
                wm = wm[:min_len]
                bf = bf[:min_len]
                bm = bm[:min_len]

                # sample 16 names from each group
                random.seed(random_state)
                wf = random.sample(wf, min(8, len(wf)))
                wm = random.sample(wm, min(8, len(wm)))
                bf = random.sample(bf, min(8, len(bf)))
                bm = random.sample(bm, min(8, len(bm)))

                formatted_names = {
                    'black': bf + bm,
                    'white': wf + wm,
                }

        if exp_framework == 'yin':
            # bf bm wf wm af am hf hm
            if id =='armstrong' or id =='zollo' or id =='yin' or id == 'gaeb':

                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                bm = names[names['group'] == 'black_male']['name'].tolist()
                bf = names[names['group'] == 'black_female']['name'].tolist()
                am = names[names['group'] == 'asian_male']['name'].tolist()
                af = names[names['group'] == 'asian_female']['name'].tolist()
                hm = names[names['group'] == 'hispanic_male']['name'].tolist()
                hf = names[names['group'] == 'hispanic_female']['name'].tolist()

                if len(wm) > 100:
                    # sample 100 names from wm
                    wm = random.sample(wm, 100) 
                if len(wf) > 100:
                    wf = random.sample(wf, 100)
                if len(bm) > 100:
                    bm = random.sample(bm, 100)
                if len(bf) > 100:
                    bf = random.sample(bf, 100)
                if len(am) > 100:
                    am = random.sample(am, 100)
                if len(af) > 100:
                    af = random.sample(af, 100)
                if len(hf) > 100:
                    hf = random.sample(hf, 100)
                if len(hm) > 100:
                    hm = random.sample(hm, 100)
                

                formatted_names = {
                    'W_W': wf, 'B_W': bf, 'A_W': af, 'H_W': hf, 'W_M': wm, 'B_M': bm, 'A_M': am, 'H_M': hm
                }
                
            if id == 'lippens':
                # Use all 18 lippens demographic-sex groups (9 demographics × 2 sexes),
                # sampling up to 100 names per group.
                lippens_groups_all = [
                    'arab_male', 'arab_female',
                    'asian_male', 'asian_female',
                    'black_american_male', 'black_american_female',
                    'central_african_male', 'central_african_female',
                    'dutch_male', 'dutch_female',
                    'eastern_european_male', 'eastern_european_female',
                    'hispanic_male', 'hispanic_female',
                    'turkish_male', 'turkish_female',
                    'white_american_male', 'white_american_female',
                ]
                lippens_groups_race = [
                    'arab', 
                    'asian', 
                    'black_american',
                    'central_african',
                    'dutch',
                    'eastern_european',
                    'hispanic',
                    'turkish',
                    'white_american',
                ]

                formatted_names = {}
                random.seed(random_state)

                # sample 4 groups from lippens_groups_race
                sampled_groups = random.sample(lippens_groups_race, 4)

                for group in sampled_groups:
                    for sex in ['male', 'female']:
                        group_name = f"{group}_{sex}"
                        group_names = names[names['group'] == group_name]['name'].tolist()
                        if len(group_names) > 100:
                            group_names = random.sample(group_names, 100)
                        
                        formatted_group_name = group
                        if sex == 'male':
                            formatted_group_name += '_M'
                        else:
                            formatted_group_name += '_F'
                        formatted_names[formatted_group_name] = group_names
                        

            # only white   
            if id =='rozado' or id =='wen' or id =='wang':
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                
                if len(wm) > 100:
                    # sample 100 names from wm
                    wm = random.sample(wm, 100) 
                if len(wf) > 100:
                    wf = random.sample(wf, 100)
                

                formatted_names = {
                    'W_W': wf, 'W_M': wm,
                }

            # only black and white 
            if id =='karvonen' or id == 'seshadri':
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                bm = names[names['group'] == 'black_male']['name'].tolist()
                bf = names[names['group'] == 'black_female']['name'].tolist()
                
                
                if len(wm) > 100:
                    # sample 100 names from wm
                    wm = random.sample(wm, 100) 
                if len(wf) > 100:
                    wf = random.sample(wf, 100)
                if len(bm) > 100:
                    bm = random.sample(bm, 100)
                if len(bf) > 100:
                    bf = random.sample(bf, 100)
                formatted_names = {
                    'W_W': wf, 'B_W': bf, 'W_M': wm, 'B_M': bm,
                }


        if exp_framework == 'wang':
            # goal for the total names: 8
            # bf bm wf wm af am hf hm
            if id =='armstrong' or id =='zollo' or id =='yin' or id == 'gaeb':

                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                bm = names[names['group'] == 'black_male']['name'].tolist()
                bf = names[names['group'] == 'black_female']['name'].tolist()
                am = names[names['group'] == 'asian_male']['name'].tolist()
                af = names[names['group'] == 'asian_female']['name'].tolist()
                hm = names[names['group'] == 'hispanic_male']['name'].tolist()
                hf = names[names['group'] == 'hispanic_female']['name'].tolist()

                if len(wm) > 1:
                    # sample 4 names from wm
                    wm = random.sample(wm, 1) 
                if len(wf) > 1:
                    wf = random.sample(wf, 1)
                if len(bm) > 1:
                    bm = random.sample(bm, 1)
                if len(bf) > 1:
                    bf = random.sample(bf, 1)
                if len(am) > 1:
                    am = random.sample(am, 1)
                if len(af) > 1:
                    af = random.sample(af, 1)
                if len(hf) > 1:
                    hf = random.sample(hf, 1)
                if len(hm) > 1:
                    hm = random.sample(hm, 1)
                

                formatted_names = {
                    'F': wf+bf+af+hf, # 4 names 
                    'M': wm+bm+am+hm # 4 names 
                }
                

            # only white   
            if id == 'lippens':
                arm = names[names['group'] == 'arab_male']['name'].tolist()
                arf = names[names['group'] == 'arab_female']['name'].tolist()
                am = names[names['group'] == 'asian_male']['name'].tolist()
                af = names[names['group'] == 'asian_female']['name'].tolist()
                bam = names[names['group'] == 'black_american_male']['name'].tolist()
                baf = names[names['group'] == 'black_american_female']['name'].tolist()
                cam = names[names['group'] == 'central_african_male']['name'].tolist()
                caf = names[names['group'] == 'central_african_female']['name'].tolist()
                dm = names[names['group'] == 'dutch_male']['name'].tolist()
                df = names[names['group'] == 'dutch_female']['name'].tolist()
                eem = names[names['group'] == 'eastern_european_male']['name'].tolist()
                eef = names[names['group'] == 'eastern_european_female']['name'].tolist()
                hm = names[names['group'] == 'hispanic_male']['name'].tolist()
                hf = names[names['group'] == 'hispanic_female']['name'].tolist()
                tm = names[names['group'] == 'turkish_male']['name'].tolist()
                tf = names[names['group'] == 'turkish_female']['name'].tolist()
                wam = names[names['group'] == 'white_american_male']['name'].tolist()
                waf = names[names['group'] == 'white_american_female']['name'].tolist()

                arm = random.sample(arm, 1)
                arf = random.sample(arf, 1)
                am = random.sample(am, 1)
                af = random.sample(af, 1)
                bam = random.sample(bam, 1)
                baf = random.sample(baf, 1)
                cam = random.sample(cam, 1)
                caf = random.sample(caf, 1)
                dm = random.sample(dm, 1)
                df = random.sample(df, 1)
                eem = random.sample(eem, 1)
                eef = random.sample(eef, 1)
                hm = random.sample(hm, 1)
                hf = random.sample(hf, 1)
                tm = random.sample(tm, 1)
                tf = random.sample(tf, 1)
                wam = random.sample(wam, 1)
                waf = random.sample(waf, 1)

                # map each demographic to its (male, female) sampled name
                demographics = {
                    'arab': (arm, arf),
                    'asian': (am, af),
                    'black_american': (bam, baf),
                    'central_african': (cam, caf),
                    'dutch': (dm, df),
                    'eastern_european': (eem, eef),
                    'hispanic': (hm, hf),
                    'turkish': (tm, tf),
                    'white_american': (wam, waf),
                }

                # pick 4 of the 9 demographics; use the same 4 for both genders
                selected = random.sample(list(demographics.keys()), 4)

                formatted_names = {
                    'F': sum((demographics[d][1] for d in selected), []),  # 4 names
                    'M': sum((demographics[d][0] for d in selected), []),  # 4 names
                }


            if id =='rozado' or id =='wen' or id =='wang':
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                
                if len(wm) > 4:
                    # sample 100 names from wm
                    wm = random.sample(wm, 4)
                if len(wf) > 4:
                    wf = random.sample(wf, 4)
                

                formatted_names = {
                    'F': wf, 'M': wm,
                }

            # only black and white 
            if id =='karvonen' or id == 'seshadri':
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                bm = names[names['group'] == 'black_male']['name'].tolist()
                bf = names[names['group'] == 'black_female']['name'].tolist()
                
                
                if len(wm) > 2:
                    # sample 2 names from wm
                    wm = random.sample(wm, 2) 
                if len(wf) > 2:
                    wf = random.sample(wf, 2)
                if len(bm) > 2:
                    bm = random.sample(bm, 2)
                if len(bf) > 2:
                    bf = random.sample(bf, 2)
                formatted_names = {
                    'F': wf+ bf, 
                    'M': wm+ bm,
                }

        return formatted_names
