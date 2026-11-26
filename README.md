# HPV vaccine acceptance among parents in Bangladesh: an ML study

CSE445 group project. **Deadline: Tuesday 15 December 2026.** We finish on Monday 14 December.

Read sections 1 to 4 first. They take ten minutes. Section 4 tells you what to do today.

---

## 1. The project in three sentences

1. The data is 2,151 parents of daughters aged 9 to 15. They were interviewed face to face in all 8 divisions of Bangladesh in 2023 (Hawlader et al., PLOS ONE 2024).
2. Our models read a parent's answers and predict one of three classes: **Yes**, **Only if free**, **Still not yes**. This is the parent's answer to "will you give your daughter the HPV vaccine?"
3. The authors tested their model on the same parents it was trained on. We test on parents the model never saw, and also on a whole division it never saw.

Learn these three sentences. Sir can ask any member.

## 2. Where we are

Updated: **Thursday 26 November 2026**. The lead updates this table after every stage.

| # | Stage | Status | Dates |
|---|---|---|---|
| 0 | Set up: the GitHub repo, and Colab working for all five | **In progress** | Thu 26 - Fri 27 Nov |
| 1 | Clean the data | **Done** | finished 26 Nov |
| 2 | Look at the data (EDA) | **Next. This is everyone's first task.** | Fri 27 - Sat 28 Nov |
| 3 | First models | Not started | Wed 2 - Sat 5 Dec |
| 4 | More models, two tests | Not started | Sat 5 - Wed 9 Dec |
| 5 | Explain the models | Not started | Wed 9 - Fri 11 Dec |
| 6 | Write the report | Not started. It grows one section at a time. | Tue 1 - Sun 13 Dec |
| 7 | Pack the zip and rehearse | Not started | Sun 13 - Mon 14 Dec |

```
set up -> clean -> EDA -> first models -> more models + division test -> explain -> pack
  (0)      (1)     (2)        (3)                  (4)                    (5)       (7)
                    the report (6) grows next to stages 3, 4 and 5
```

What exists today:

| File | What it is |
|---|---|
| `data/idea2_hpv_vaccine_bangladesh/s003.csv` | The raw file from the paper: 2,151 rows, 131 columns. **Never edit it.** |
| `data/hpv_clean.csv` | **The one clean file. Every notebook reads this file.** 2,147 rows, 101 columns. |
| `notebooks/01_clean.ipynb` | How the clean file was made, with the reason for every decision |
| `checks/sanity_check_clean_file.py` | 16 checks that compare the clean file with the raw file. All pass. |

## 3. Words we use

| Word | Meaning |
|---|---|
| Row | One parent |
| Column | One question of the survey, or something made from a question |
| Feature | A column the model is allowed to read |
| Target | The column the model must predict. Ours is `target_3class`. |
| Class | One of the possible answers in the target: Yes, Only if free, Still not yes |
| Notebook | A file of code and text that you run piece by piece. Its name ends in `.ipynb`. |
| Cell | One piece of a notebook. You run it with Shift + Enter. |
| EDA | Exploratory data analysis. Looking at the data with tables and charts before building any model. |
| Baseline | The score you get with no model, by always guessing the most common class. Ours is 70.3%. |
| Train set, test set | The parents the model learns from, and the parents kept hidden to test it |
| Split | The list that says which parent is in the train set and which is in the test set |

The 101 columns of the clean file:

| Kind | How many | Rule |
|---|---|---|
| Features | 97 | Every column that does not start with `target_` or `meta_` |
| Targets | 2 | `target_3class` and `target_2class`. Never a feature. |
| Meta | 2 | `meta_division` and `meta_district`. Never a feature. Only for charts and for the division test. |

`target_3class` has: **Yes 1,509** (70.3%), **Only if free 354** (16.5%), **Still not yes 284** (13.2%).

## 4. What to do right now

### If you are a member

Do these in order. Stop and ask the lead if a step does not work.

| Step | What to do | Done when |
|---|---|---|
| 1 | Make a free account on [github.com](https://github.com). Send your GitHub username to the lead. | The lead has your username |
| 2 | Open the invitation email from GitHub and click **Accept invitation**. | You can open this repo in your browser |
| 3 | Do the Colab test in section 5 | You see `(2147, 101)` on your screen |
| 4 | Do your EDA task in section 6 | Your notebook is in the `notebooks` folder of this repo by **Saturday 28 November, 11 PM** |

### If you are the lead

| Step | What to do |
|---|---|
| 1 | Put this repo on GitHub |
| 2 | Add the four members to the repo (Settings > Collaborators > Add people) |
| 3 | Explain sections 1 to 6 to the team in one call |
| 4 | Do your own EDA task, like every member |
| 5 | On Sunday 29 November: open and run all five EDA notebooks, then give the roles for stages 3 to 7 |
| 6 | Before Wednesday 2 December: make the one split and put it in `data/split.csv` |
| 7 | After every stage: run the finished notebooks from top to bottom, and update the table in section 2 |

## 5. The Colab test (5 minutes, do it once)

Google Colab runs notebooks in your browser. You install nothing.

1. Open [colab.research.google.com](https://colab.research.google.com) and sign in with your Google account.
2. Click **File > New notebook in Drive**. An empty notebook opens with one empty cell.
3. Click inside the cell. Paste these three lines:
   ```python
   import pandas as pd
   df = pd.read_csv("https://raw.githubusercontent.com/SumayerKhan/cse445-hpv-vaccine-ml/main/data/hpv_clean.csv")
   df.shape
   ```
4. Press **Shift + Enter**. Wait a few seconds.
5. Under the cell you must see: `(2147, 101)`. That means 2,147 rows and 101 columns. The clean file is loaded.
6. Click **File > Save a copy in GitHub**. The first time, a window asks you to sign in to GitHub and to allow Colab. Allow it.
7. In the box **Repository**, choose `SumayerKhan/cse445-hpv-vaccine-ml`. In the box **File path**, type `notebooks/00_test_yourname.ipynb` with your own name. Click **OK**.
8. Open the repo in your browser. Your file must be in the `notebooks` folder.

If step 5 shows an error, or step 8 shows no file, send a screenshot to the lead.

## 6. Stage 2, everyone's first task: EDA of one feature group

**Due: Saturday 28 November, 11 PM.** File name: `notebooks/02_eda_yourname.ipynb`.

### What EDA is, and why we do it

Before we build a model, we look at the data. For each feature we ask one question: **do parents who gave different answers to this question also differ in the target?**

If yes, the feature can help a model. If no, it cannot. Sir asked us to understand the data and to know which factors matter. This task is that understanding. There is no model in this task.

### Your feature group

There are 97 features. We split them into five groups. Each member takes one.

| Member | Group | Features | Your notebook |
|---|---|---|---|
| Sajid (lead) | D. Other knowledge, and cues to action | 15 | `notebooks/02_eda_sajid.ipynb` |
| Punno | A. The parent and the family | 22 | `notebooks/02_eda_punno.ipynb` |
| Rio | B. HPV and the HPV vaccine: heard of it, and from where | 20 | `notebooks/02_eda_rio.ipynb` |
| Udoy | C. Cervical cancer and its vaccine: heard of it, and from where | 20 | `notebooks/02_eda_udoy.ipynb` |
| Avishekh | E. Beliefs | 20 | `notebooks/02_eda_avishekh.ipynb` |

Copy the list of your own group into your notebook:

```python
group_A = ["parent", "age", "marital_status", "religion", "residence", "education_years",
           "educ_spouse", "occupation", "occup_spouse", "hcw", "income", "family_member",
           "children_0", "daugther_0", "daughter_study", "ins_type", "ins_location",
           "religion_imp", "routine_health", "epi", "daughter_age_youngest", "daughter_age_oldest"]

group_B = ["hpv_heard", "hpv_hcw", "hpv_training", "hpv_friend", "hpv_relatives", "hpv_school",
           "hpv_pharmacy", "hpv_advert", "hpv_social", "hpv_mass",
           "vac_hear", "vac_hcw", "vac_train", "vac_friend", "vac_relative", "vac_school",
           "vac_pharmacy", "vac_advert", "vac_social", "vac_mass"]

group_C = ["cancer_hear", "cancer_hcw", "cancer_training", "cancer_friend", "cancer_relative",
           "cancer_school", "cancer_pharmacy", "cancer_advert", "cancer_social", "cancer_mass",
           "canvac_hear", "canvac_hcw", "canvac_training", "canvac_friends", "canvac_relative",
           "canvac_school", "canvac_pharmacy", "canvac_advert", "canvac_social", "canvac_mass"]

group_D = ["cancer_female", "pap_detect", "pap_done", "hpv_cervical", "hpv_transmit", "hpv_sexual",
           "hpv_blood", "hpv_pubtoilet", "hpv_mtc", "vac_reduce_risk", "recom_age",
           "cu1", "cu2", "cu3", "cu4"]

group_E = ["vul1", "vul2", "vul3", "vul4", "vul5", "sev1", "sev2", "sev3", "sev4", "sev5",
           "ben1", "ben2", "ben3", "ben4", "ben5", "bar1", "bar2", "bar3", "bar4", "bar5"]
```

To know what each column means, open the questionnaire: `data/idea2_hpv_vaccine_bangladesh/HPV_questionnaire_S1_File.docx`. The columns follow the order of the questions in it.

### The steps

**Step 1. Make the notebook.** In Colab: **File > New notebook in Drive**. Click the name at the top left and change it to `02_eda_yourname.ipynb`.

**Step 2. Load the clean file.** Paste the three lines from section 5 into the first cell and run it. You must see `(2147, 101)`.

**Step 3. Make one table for one feature.** This is the heart of the task, so here it is with a real example. The feature is `hpv_heard`: "Have you heard of HPV?"

```python
pd.crosstab(df["hpv_heard"], df["target_3class"])
```

It gives the counts:

| hpv_heard | Yes | Only if free | Still not yes |
|---|---|---|---|
| No | 1,053 | 337 | 272 |
| Yes | 456 | 17 | 12 |

Counts are hard to compare, because 1,662 parents said No and only 485 said Yes. So we also ask for the share inside each row:

```python
(pd.crosstab(df["hpv_heard"], df["target_3class"], normalize="index") * 100).round(1)
```

| hpv_heard | Yes | Only if free | Still not yes |
|---|---|---|---|
| No | 63.4 | 20.3 | 16.4 |
| Yes | 94.0 | 3.5 | 2.5 |

**How to read it.** Each row adds up to 100. The first row says: of the parents who had **not** heard of HPV, 63.4% answered Yes. The second row says: of the parents who **had** heard of HPV, 94.0% answered Yes.

**What to compare it with.** A number alone says little. Is 63.4% high or low? To know, we need the same shares for **all 2,147 parents together**, with no feature at all:

```python
(df["target_3class"].value_counts(normalize=True) * 100).round(1)
```

It gives: Yes 70.3, Only if free 16.5, Still not yes 13.2. This is our reference line. Put it under the table:

| | Yes | Only if free | Still not yes |
|---|---|---|---|
| Had not heard of HPV | 63.4 | 20.3 | 16.4 |
| Had heard of HPV | 94.0 | 3.5 | 2.5 |
| **All parents together** | **70.3** | **16.5** | **13.2** |

Now compare each row with the last row:

- Parents who had heard of HPV: 94.0% Yes. That is 23.7 points **above** all parents together.
- Parents who had not heard: 63.4% Yes. That is 6.9 points **below**.

So if you know a parent's answer to this one question, you can guess their decision better than without it. That is what "this feature is linked to the target" means.

If every row of a feature is close to the last row, the feature tells us nothing: knowing the answer does not help the guess. Always look at all three columns. A feature can be flat in the Yes column and still differ in the other two.

**Step 4. Do the same for every feature in your group.** Make both tables for each one. You can ask the AI to write a loop that does it for your whole list (see the prompt below).

**Step 5. Number features need a different table.** Nine features are numbers, not words: `age`, `education_years`, `educ_spouse`, `income`, `family_member`, `children_0`, `daugther_0`, `daughter_age_youngest`, `daughter_age_oldest`. A table with one row for each age would be too long. For these, compare the middle value in each class:

```python
df.groupby("target_3class")["age"].describe()
```

Look at the column `50%`. That is the median, the middle value. If the median is nearly the same in the three classes, the feature tells us little.

Only group A has number features.

**Step 6. Make three or four charts.** Choose the features of your group where the rows differ the most from each other. For each one make a bar chart of the shares, so the difference can be seen in one look. Ask the AI for "a stacked bar chart of the row percentages".

**Step 7. Write down anything strange.** For example: a value that almost nobody chose, two columns that look the same, a result that is the opposite of what you expected. Do not hide it and do not fix it. Write it in a text cell.

**Step 8. Write five lines in your own words.** In a text cell at the top of the notebook, answer: **what are the three strongest findings in your group?** Use numbers from your tables. This part must not be written by an AI. Sir can ask you about it.

**Step 9. Save to GitHub.** **File > Save a copy in GitHub**, repository `SumayerKhan/cse445-hpv-vaccine-ml`, file path `notebooks/02_eda_yourname.ipynb`.

### What you hand in

One notebook with: both tables for every feature in your group, three or four charts, your notes on strange things, and your five lines.

### How you know it is right

- The counts in every count table add up to 2,147.
- Every row of a percentage table adds up to 100.
- The notebook runs from the first cell to the last without an error (**Runtime > Restart session and run all**).
- You can explain any table in it without reading.

### What to tell the AI for this task

Paste the general prompt from section 12 first. Then paste this, with your own group letter and list:

> My task is EDA of one feature group. Here is my list of features: [paste your list]. For each feature in the list, show two tables: the crosstab of the feature against `target_3class`, and the same crosstab as row percentages rounded to one decimal. For numeric features, show `describe()` grouped by `target_3class` instead. Then draw a stacked bar chart of the row percentages for the features I name. Do not build any model.

## 7. Stage 3: first models (Wed 2 - Sat 5 Dec)

**Who does it:** the roles are given on Sunday 29 November.

### What it is

We build the two simplest models, so that every later model has something to be compared with.

| Model | What it does | Why we need it |
|---|---|---|
| Baseline | Always says "Yes", the most common class | It gets 70.3% with no learning. A real model must beat it. |
| Logistic regression | The method the authors used | The authors tested it on the parents it was trained on. We test it on parents it never saw. This is sentence 3 of our project. |

### What you start with

- `data/hpv_clean.csv`
- `data/split.csv`, made by the lead. It has one line for each parent and says `train` or `test`. **Everyone uses this file. Nobody makes a split.**

### The steps

1. Load the clean file and the split file. Make four pieces: the features of the train parents, their target, the features of the test parents, their target.
2. Features are all columns that do not start with `target_` or `meta_`.
3. Build the baseline with scikit-learn's `DummyClassifier(strategy="most_frequent")`. Train it on the train parents. Test it on the test parents.
4. Build the logistic regression inside a scikit-learn `Pipeline`. The pipeline must do three things in this order: fill the empty cells, turn the word answers into numbers, then fit the model. The pipeline learns all three from the train parents only.
5. For both models report, on the test parents: accuracy, macro-F1, and the confusion matrix.
6. Do the same once more with `target_2class`, because that is the target the authors used. This lets us put our test score next to their number.

### Three scores, in plain words

| Score | Meaning |
|---|---|
| Accuracy | Out of 100 test parents, how many did the model get right |
| Macro-F1 | A score that treats the three classes as equally important. Accuracy can look good while the model fails on the two small classes. Macro-F1 shows that. |
| Confusion matrix | A table: the rows are the real class, the columns are what the model said. It shows which class is mistaken for which. |

### What you hand in

`notebooks/03_first_models_yourname.ipynb`, with a small table: model, accuracy, macro-F1. And your five lines.

### How you know it is right

- The baseline's accuracy is close to 70%. If it is not, the split or the target is wrong.
- The test parents were never used for training, filling or encoding.

## 8. Stage 4: more models, and two tests (Sat 5 - Wed 9 Dec)

### What it is

We try several kinds of models, and we test each one in two ways.

| Test | How | What it answers |
|---|---|---|
| Test 1: the 80/20 split | Train on 80% of the parents, test on the other 20%, from all divisions mixed | Does the model work on new parents from the same places? |
| Test 2: leave one division out | Train on seven divisions, test on the eighth. Repeat eight times, so each division is the test once. | Does the model work in a place it has never seen? |

### Why test 2 matters

The share of parents who said Yes differs a lot between divisions. Each area had its own interviewers. So a model may learn "who asked the question" in place of "what the parent thinks". Test 2 catches this. If a model is good in test 1 and bad in test 2, we report it openly. This test is our own addition, and it is the strongest part of the project.

### The steps

1. Keep the pipeline from stage 3. Only the last part, the model, changes.
2. Models to try: decision tree, random forest, gradient boosting, k-nearest neighbours, support vector machine, naive Bayes. Each member who has this role takes one or two.
3. Run test 1 with the split file.
4. Run test 2 with scikit-learn's `LeaveOneGroupOut`, with `meta_division` as the groups.
5. Run both tests two times: with all features, and without the 20 belief features (`vul1` to `bar5`). Many parents had never heard of HPV, so their belief answers may mean little.
6. Put every score in one table.

### What you hand in

`notebooks/04_models_yourname.ipynb`, and your rows for the one shared score table:

| Model | Features | Test 1 accuracy | Test 1 macro-F1 | Test 2 accuracy | Test 2 macro-F1 |
|---|---|---|---|---|---|

## 9. Stage 5: explain the models (Wed 9 - Fri 11 Dec)

### What it is

A score is not enough. Sir said accuracy alone is not the important thing. We must say what the model uses, and where it fails.

### The steps

1. Take the best model from stage 4.
2. **Which features matter:** use permutation importance. It shuffles one feature and measures how much the score falls. A large fall means the model needs that feature.
3. **Where the model fails:** read the confusion matrix. Which class does it miss most? We expect the two small classes to be the hard ones.
4. **Division by division:** show the score of test 2 for each of the eight divisions.
5. Compare with stage 2. Do the features the model uses match what we saw in the EDA?

### What you hand in

`notebooks/05_explain_yourname.ipynb`: one chart of the ten most important features, one confusion matrix, one table of the eight divisions, and your five lines.

## 10. Stage 6: the report (Tue 1 - Sun 13 Dec)

The report is written while the work is done, not at the end. Each section has an owner. The owners are named on Sunday 29 November.

| Section | What goes in it | Written after |
|---|---|---|
| 1. Introduction | What HPV and the vaccine are, why it matters in Bangladesh, our question | Now |
| 2. Related work | The paper we take the data from, and what other studies did | Now |
| 3. Data | The survey, the 2,151 parents, what we cleaned and why | Stage 2 |
| 4. Method | Features, targets, the pipeline, the models, the two tests | Stage 4 |
| 5. Results | The score table, the charts | Stage 4 and 5 |
| 6. Discussion and limits | What the results mean, the division problem, what we cannot claim | Stage 5 |
| 7. Conclusion | Half a page | Last |

Three rules for the report:

1. Say "we did not find", never "nobody has done this".
2. Say "linked to", never "causes". A survey taken once cannot show a cause.
3. Every score stands next to the baseline.

## 11. Stage 7: pack and rehearse (Sun 13 - Mon 14 Dec)

The zip must hold:

1. The report as Word or LaTeX
2. The report as PDF
3. One Jupyter notebook, joined by the lead from all parts, that runs from top to bottom
4. The raw data: `s003.csv`
5. A `.txt` file with the group number and each member's name, ID and phone

Then each member explains their own part aloud to the group, once, without reading.

## 12. What to tell the AI

You may use ChatGPT or Claude to write the code. Paste this first, every time, and then your task:

> I am working on a group ML project in Google Colab with pandas and scikit-learn. The data is `hpv_clean.csv`: 2,147 parents in Bangladesh, 101 columns, from a survey about the HPV vaccine. The target is `target_3class` with the classes "Yes", "Only if free" and "Still not yes". Columns that start with `target_` or `meta_` must never be used as features. Most features are text categories such as Yes/No or Agree/Disagree. Nine features are numeric. Do not clean the data again, do not drop rows, and do not make a new train/test split. Explain each cell in one or two simple sentences.

Then check what the AI gives you against the real file. The AI can be wrong about the data. **You are responsible for your part, not the AI.**

## 13. The rules

1. **One clean file.** Nobody makes their own cleaned copy. If you think the clean file is wrong, tell the lead.
2. **One split.** The lead provides it. Scores from different splits cannot be compared.
3. **Your own notebook only.** Never edit another member's notebook.
4. **Never use a `target_` or `meta_` column as a feature.**
5. **Every score is reported next to the baseline, 70.3%.**
6. **Filling empty cells, encoding and scaling are learned from the train parents only**, inside a scikit-learn `Pipeline`.
7. **Five lines in your own words** for every part you hand in.
8. **Every part has a date.** We check for 10 minutes after each CSE445 class, on Monday and Wednesday. If you will be late, say so before the date, not after.

## 14. Source of the data

Hawlader MDH et al. (2024). Acceptance of Human Papillomavirus (HPV) vaccine among the parents of eligible daughters (9-15 years) in Bangladesh: A nationwide study using Health Belief Model. PLOS ONE 19(11): e0310779. https://doi.org/10.1371/journal.pone.0310779
