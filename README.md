# HPV vaccine acceptance among parents in Bangladesh: an ML study

CSE445 group project. **We hand in on Sunday 20 December 2026.** We finish on Thursday 10 December.

Sections 1 to 4 take ten minutes to read. Section 4 says what each of us does now.

---

## 1. The project in three sentences

1. We have the answers of 2,147 parents of daughters aged 9 to 15, interviewed in all 8 divisions of Bangladesh in 2023 (Hawlader et al., PLOS ONE 2024).
2. From a parent's answers, our models predict one of three classes: **Will vaccinate**, **Only if free**, **Undecided or no**.
3. The authors tested their model on the same parents it was trained on. We test on parents the model never saw, and on a whole division it never saw.

We all learn these three sentences. Sir can ask any of us.

**What is new in our project.** Our main model is a Bayesian model. A classical model learns one value for each weight. A Bayesian model learns a range of values for each weight: a narrow range where we have many parents, a wide range where we have few. So it can say how sure it is, and one model that we fit once can answer many questions about the decision. The classical models are what we compare it with. The authors used logistic regression, and we did not find a Bayesian model on this data.

## 2. Where we are

Updated: **Saturday 28 November 2026**.

| # | Stage | Status | Dates |
|---|---|---|---|
| 1 | Clean the data | **Done** | 26 Nov |
| 2 | Look at the data (EDA) | **Done** | 27 Nov |
| 3 | Dataset guide and the one split | **Done** | 27 Nov |
| 4 | Train the models | **Now. One notebook for each of us.** | 28 Nov - 2 Dec |
| 5 | Compare and explain | Not started | 2 - 7 Dec |
| 6 | Write the report | Not started | 2 - 10 Dec |
| 7 | Pack the zip and rehearse | Not started | 10 - 19 Dec |

What exists today:

| File | What it is |
|---|---|
| `data/raw/s003.csv` | The raw file from the paper: 2,151 rows, 131 columns. **We never edit it.** |
| `data/raw/` (other files) | The paper, the questionnaire, and the table of the sample by division |
| `data/hpv_clean.csv` | **The one clean file. Every notebook reads this file.** 2,147 rows, 101 columns. |
| `data/split.csv` | **The one split.** 1,717 train parents, 430 test parents. |
| `docs/dataset_guide.md` | **Read this first.** What is in the data, what we found, and the rules for the models. |
| `notebooks/01_clean.ipynb` | How we made the clean file, with the reason for every decision |
| `notebooks/02_eda.ipynb` | Our EDA of all 97 features |
| `notebooks/03_split.ipynb` | How we made the split |
| `notebooks/04_models_<name>.ipynb` | One notebook for each of us, with gaps to fill in |
| `checks/sanity_check_clean_file.py` | 16 checks that compare the clean file with the raw file. All pass. |

How the repository is organised:

```
README.md              this file
requirements.txt       the Python packages we use
data/
  raw/                 the files from the paper: s003.csv, the paper, the questionnaire
  hpv_clean.csv        the one clean file
  split.csv            the one split
docs/
  dataset_guide.md     what is in the data, what we found, the rules
notebooks/
  01_clean.ipynb       how we made the clean file
  02_eda.ipynb         our EDA
  03_split.ipynb       how we made the split
  04_models_<name>.ipynb   one for each of us
results/               the result file of each model notebook goes here
checks/                a program that checks the clean file against the raw file
presentation/          the slides of our plan
```

## 3. Words we use

| Word | Meaning |
|---|---|
| Row | One parent |
| Feature | A column the model is allowed to read. We have 97. |
| Target | The column the model must predict. Ours is `target_3class`. |
| Class | One of the three values of the target |
| Baseline | The score with no model, by always saying the most common class. Ours is 70.3%. |
| Train set, test set | The parents the model learns from, and the parents kept hidden to test it |
| Split | The file that says which parent is in the train set and which is in the test set |
| Settings | The choices we make for a model before it learns, for example how deep a tree may grow |
| Frame | The cells of a notebook that are already written |
| Gap | A cell we fill in ourselves |

The three classes:

| Class | Parents | Share |
|---|---|---|
| **Will vaccinate** | 1,509 | 70.3% |
| **Only if free** | 354 | 16.5% |
| **Undecided or no** | 284 | 13.2% |

## 4. Who does what

### The models

| Member | Models | Notebook |
|---|---|---|
| Punno | Gradient boosting | `notebooks/04_models_punno.ipynb` |
| Rio | Decision tree, random forest | `notebooks/04_models_rio.ipynb` |
| Udoy | k-nearest neighbours, naive Bayes | `notebooks/04_models_udoy.ipynb` |
| Avishek | Support vector machine | `notebooks/04_models_avishek.ipynb` |
| Sajid | Logistic regression, and the Bayesian model | `notebooks/04_models_sajid.ipynb` |

### One more task for each of us

| Member | Task |
|---|---|
| Punno | Join the five result files into one comparison table, and make its charts |
| Rio | Which features matter most, from the tree models |
| Udoy | Report: introduction, the data, related work |
| Avishek | The final slides, the zip, and the text file with our names, IDs and phones |
| Sajid | The Bayesian model. Report: method and limits. Final check of everything. |

Each of us also writes the report section about our own model: how it works, its results, one table.

### Dates

| Date | What is due |
|---|---|
| **Sunday 29 November** | The notebook runs to "All checks passed", with any settings |
| **Wednesday 2 December** | Final notebook: good settings, GAP 2 and GAP 3 written |
| **Monday 7 December** | Our report section, and our extra task |
| **Thursday 10 December** | Everything is joined into one report |
| Sunday 20 December | We hand in |

## 5. The model notebook

Every notebook has the same frame and three gaps.

| Part | What it does |
|---|---|
| Frame 1 to 3 | Loads the data and the split, picks the features, turns the answers into numbers |
| **GAP 1** | **The model, and the settings to try** |
| Frame 4 to 6 | Trains, runs both tests, shows the tables, checks, saves the result file |
| **GAP 2** | **Five lines: how the model works** |
| **GAP 3** | **Five lines: what the results show** |

The frame is the same for all of us, so that all results can be compared: the same data, the same split, the same features, the same scores.

### The steps

1. Read `docs/dataset_guide.md`.
2. Open the notebook. In Google Colab: **File > Open notebook > GitHub**, repository `SumayerKhan/cse445-hpv-vaccine-ml`.
3. Fill in GAP 1: the model, and at least two settings with three values each.
4. Run the whole notebook: **Runtime > Restart session and run all**.
5. The last code cell must print **All checks passed**. If it does not, the error says what is wrong.
6. Read the tables. Write GAP 2 and GAP 3 in our own words, with numbers from the tables.
7. Hand in two files: the notebook, and `results/04_models_<name>.csv`. In Colab the result file is in the file panel on the left, in the folder `results`. Download it.
8. Put the notebook in `notebooks/` of this repo (**File > Save a copy in GitHub**), and the result file in `results/`. If that fails, send both files to Sajid.

### The two tests the frame runs

| Test | How | What it answers |
|---|---|---|
| The 80/20 split | Train on the 1,717 train parents, test on the 430 test parents | Does the model work on new parents from the same places? |
| Leave one division out | Train on seven divisions, test on the eighth. Eight times. | Does the model work in a place it has never seen? |

Every model is run twice: without the five barrier features (`bar1` to `bar5`), and with them. The dataset guide, section 5.5, says why.

### The scores

| Score | Meaning |
|---|---|
| Accuracy | Out of 100 test parents, how many the model got right |
| Macro-F1 | A score that treats the three classes as equally important. Accuracy can look good while the model fails on the two small classes. |
| Right in each class | Of the parents who are really in a class, the share the model put there |
| Baseline accuracy | What a model gets that always says the most common class |

## 6. The rules

1. **Fill in the gaps only.** We do not change any other cell.
2. **One clean file, one split.** Nobody loads other data, makes a split, or chooses features. The frame does this.
3. **The notebook runs from top to bottom** and prints "All checks passed".
4. **We can explain our own model.** Any tool may help with the code. GAP 2 and GAP 3 are our own words, because sir can ask any of us.
5. **Our own notebook only.** We never edit another member's notebook.
6. **Every score stands next to the baseline.**
7. **Stuck for more than 30 minutes: write in the group.** Late: say so before the date, not after.

## 7. The report

| Section | Owner |
|---|---|
| 1. Introduction | Udoy |
| 2. Related work | Udoy |
| 3. Data | Udoy, from the dataset guide |
| 4. Method | Sajid |
| 5. Results: one part for each model | Each of us |
| 5. Results: the comparison table and charts | Punno |
| 5. Results: which features matter | Rio |
| 6. Discussion and limits | Sajid |
| 7. Conclusion | Sajid |

Three rules for the report:

1. We say "we did not find", never "nobody has done this".
2. We say "linked to", never "causes". A survey taken once cannot show a cause.
3. Every score stands next to the baseline.

## 8. The zip

1. The report as Word or LaTeX
2. The report as PDF
3. The Jupyter notebooks
4. The raw data: `s003.csv`
5. A `.txt` file with the group number and each member's name, ID and phone

## 9. Source of the data

Hawlader MDH et al. (2024). Acceptance of Human Papillomavirus (HPV) vaccine among the parents of eligible daughters (9-15 years) in Bangladesh: A nationwide study using Health Belief Model. PLOS ONE 19(11): e0310779. https://doi.org/10.1371/journal.pone.0310779

The paper and its data files are published under the Creative Commons Attribution License (CC BY 4.0). It allows use and sharing when the authors and the source are credited. The files in `data/raw/` are the authors' work. We did not collect this data.
