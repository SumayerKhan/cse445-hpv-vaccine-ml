# Dataset guide

This guide is for every member of the group. Read it before you train a model.

It says what the data is, what we found in it, and the rules that follow for the models. Every number comes from `notebooks/01_clean.ipynb` and `notebooks/02_eda.ipynb`.

---

## 1. The data in six facts

| | |
|---|---|
| Who | 2,151 parents of daughters aged 9 to 15, in all 8 divisions of Bangladesh |
| How | Face-to-face interviews, 28 June to 2 August 2023 |
| Source | Hawlader et al., PLOS ONE, 2024. Most authors are from North South University. |
| Raw file | `data/raw/s003.csv`: 2,151 rows, 131 columns |
| Clean file | `data/hpv_clean.csv`: **2,147 rows, 101 columns**. Every notebook reads this file. |
| What a row is | One parent. Each column is one question of the survey. |

## 2. The 101 columns

| Kind | How many | Rule |
|---|---|---|
| Features | 97 | Every column that does not start with `target_` or `meta_` |
| Targets | 2 | `target_3class` (our main target) and `target_2class` (the authors' target). Never a feature. |
| Meta | 2 | `meta_division`, `meta_district`. Never a feature. Only for charts and for the division test. |

The 97 features, in five groups:

| Group | Features | What the questions are about |
|---|---|---|
| A. The parent and the family | 22 | Mother or father, age, schooling, job, income, family, the daughter's school, health check-ups. 9 of these are numbers. |
| B. HPV and the HPV vaccine | 20 | Has the parent heard of HPV, and of the vaccine. From where: doctor, friends, school, TV and so on. |
| C. Cervical cancer and its vaccine | 20 | The same two questions and the same sources, for cervical cancer |
| D. Other knowledge, and cues to action | 15 | Facts about cervical cancer, the Pap test and HPV. Cancer in the parent's own history or in the family. |
| E. Beliefs | 20 | 20 statements, answered from "Strongly disagree" to "Strongly agree" |

Almost all features are categories written as words: "Yes", "No", "Agree". The question behind each column is in `data/raw/HPV_questionnaire_S1_File.docx`. The columns follow the order of the form.

## 3. The target

`target_3class` is built from two questions: "Will you give your daughter the HPV vaccine?" and "...if the government provided it for free?"

| Class | Parents | Share | Who is in it |
|---|---|---|---|
| Will vaccinate | 1,509 | 70.3% | Said yes to the first question |
| Only if free | 354 | 16.5% | Did not say yes to the first, but said yes if it is free |
| Undecided or no | 284 | 13.2% | Did not say yes to either. 267 said "not sure" to both. 17 said "no" at least once. |

**Baseline: 70.3%.** A model that always says "Will vaccinate" is right for 70.3% of the parents. A model at 70% has learned nothing.

The target is what a parent **says** they will do. It is not what they did.

## 4. What we changed when we cleaned the data

| What | Detail |
|---|---|
| Typing mistakes | In the published file "No" had become "0" inside words, for example "Don't K0w". We fixed them. |
| Rows | We dropped 3 parents with no answer to the target questions, and 1 duplicate row |
| Wrong values | 126 values of `educ_spouse` were calendar years, not years of schooling. We set them to missing. |
| Columns | We dropped 36 columns: the columns that are the target, exact copies, regroupings of other columns, and free text |
| Empty cells from skipped questions | A parent who never heard of HPV was not asked "from where". We filled those with "No". |

All of it, with the reason for each decision, is in `notebooks/01_clean.ipynb`. A separate program, `checks/sanity_check_clean_file.py`, compares the clean file with the raw file. All 16 checks pass.

## 5. What we found in the data

### 5.1 Knowledge is the strongest group of features

| Parents who... | Will vaccinate | Parents |
|---|---|---|
| have heard of HPV | 94.0% | 485 |
| know how HPV is transmitted | 97.4% | 378 |
| know that a Pap test helps early detection | 94.8% | 714 |
| are health workers | 98.4% | 183 |
| have never heard of cervical cancer | 42.3% | 426 |
| All parents | 70.3% | 2,147 |

### 5.2 Many columns say the same thing

- `hpv_heard` and `vac_hear`: 2,072 of 2,147 parents gave the same answer to both.
- `hpv_transmit` and `hpv_sexual`: they differ for only 10 parents.
- The 36 "from where did you hear" columns were only asked after a Yes to "have you heard of it". A Yes in any of them first of all means "has heard of it".

### 5.3 The divisions are very different

| Division | Parents | Will vaccinate | Only if free | Undecided or no |
|---|---|---|---|---|
| Chattogram | 433 | 48.3 | 32.1 | 19.6 |
| Rajshahi | 265 | 62.3 | 32.5 | 5.3 |
| Dhaka | 560 | 63.0 | 11.6 | 25.4 |
| Barishal | 127 | 69.3 | 15.0 | 15.7 |
| Khulna | 231 | 79.7 | 15.2 | 5.2 |
| Sylhet | 143 | 90.9 | 2.8 | 6.3 |
| Rangpur | 228 | 97.4 | 2.2 | 0.4 |
| Mymensingh | 160 | 98.8 | 0.6 | 0.6 |

Each area had its own interviewers. So part of these differences may come from how the interviewers asked the questions or wrote down the answers. We cannot check this from the data.

### 5.4 A result for all parents together can come from one or two divisions

Example: parents who answered "No" to "Does cervical cancer only occur in females?". Of all 322 of them, 47.5% are in "Undecided or no". Inside the divisions:

| Division | Parents who answered "No" | Of them, in "Undecided or no" | Parents who answered "Yes" | Of them, in "Undecided or no" |
|---|---|---|---|---|
| Dhaka | 131 | 106 (81%) | 429 | 36 (8%) |
| Chattogram | 67 | 38 (57%) | 366 | 47 (13%) |
| Khulna | 46 | 1 (2%) | 185 | 11 (6%) |

The result is there in Dhaka and Chattogram, and not in Khulna. Results that hold in three or four divisions: having heard of HPV, knowing about the Pap test, and the daughter going to a madrasa.

### 5.5 The five barrier statements contain the decision

| Feature | Statement |
|---|---|
| `bar1` | I shall not vaccinate my daughter as it is painful |
| `bar2` | I shall not vaccinate my daughter as the HPV vaccine can cause an adverse effect |
| `bar3` | I shall not vaccinate my daughter as the HPV vaccine needs two injections |
| `bar4` | The HPV vaccine is so new that I want to wait a while before deciding if my daughter should get it |
| `bar5` | I shall vaccinate my daughter even if it is not free despite knowing that the HPV vaccine cost around 2500 taka |

The target is the answer to "Will you give your daughter the HPV vaccine?". These statements ask almost the same thing. A model that reads them is partly reading the answer. This is target leakage.

### 5.6 "Neutral" often means "I do not know"

1,662 parents had never heard of HPV, and they were still asked the 20 belief statements. Most of them answered "Neutral". 166 parents gave the same answer to all 20 statements.

## 6. Rules for everyone who trains a model

These rules come from what we found. We decided them on 27 November.

| # | Rule | Why |
|---|---|---|
| 1 | Features are the columns that do not start with `target_` or `meta_` | The targets are the answer. The division and district must not be learned. |
| 2 | **Leave out `bar1` to `bar5` in the main models.** We run each model once more with them and report both scores. | Target leakage (5.5) |
| 3 | Use the one split in `data/split.csv`. Do not make your own. | Scores from different splits cannot be compared |
| 4 | Turn every category into numbers with one-hot encoding, inside a scikit-learn `Pipeline` | The encoder must learn from the training set only |
| 5 | Treat the belief answers as five separate categories, not as the numbers 1 to 5 | "Neutral" is not the middle for a parent who has never heard of HPV (5.6) |
| 6 | Merge very small categories: `marital_status` into "Living with a spouse" and "Other". `religion` into Islam, Hindu, Other. "Student" into "Others" for both occupation columns. | A model cannot learn from 4 parents |
| 7 | Fill a missing number with the median and a missing category with the most common one, learned from the training set. Do **not** add a column that marks which values were missing. | 109 of the 126 missing values of `educ_spouse` are from Rajshahi. Such a column would tell the model the division. |
| 8 | For logistic regression, use the logarithm of `income` | One parent has an income of 1,000,000 taka |
| 9 | Report every score next to the baseline, 70.3% | So we can see what the model learned |
| 10 | Report every result for all parents together and inside each division | A result can come from one or two divisions only (5.4) |
| 11 | Columns that are almost copies stay in the model. But never report two of them as two findings. | See 5.2 |

## 7. What this data cannot tell us

- **Intention, not action.** We predict what parents say they will do.
- **Not a random sample.** The paper says the parents were chosen by convenience, which means those who were easy to reach.
- **Interviewers.** Differences between divisions may partly come from the interviewers.
- **No causes.** The survey was taken once. We can say that two things are linked, not that one causes the other.
- **49 parents** have no daughter aged 9 to 15 in the file, although the study was about such parents. We kept them.
