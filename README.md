# student-life-procrastination-detection-

# Procrastination Detection from Phone Sensor Data — StudentLife Capstone

This is my capstone project. I used the StudentLife dataset from Dartmouth (the one where they tracked 48-49 students for a term using their phones) to see if you can actually predict when a student is about to procrastinate on something, just from how their phone behaves — not from asking them directly.

## Why this dataset / why this problem

Most people who've used StudentLife before used it to predict GPA or stress levels, or to guess how many deadlines someone has coming up. Nobody really used it to look at procrastination itself — the actual behavior of putting things off until the last minute. Separately, there's a decent amount of research on procrastination prediction, but it's almost always based on either surveys (asking students how much they procrastinate) or on LMS data like when they submit assignments. None of that uses actual phone sensor data. So that's the gap I went after — combining the sensor side of StudentLife with the idea of detecting procrastination as a behavior pattern.

## What's in here

- The main Jupyter/Colab notebook with the full pipeline (data merging, cleaning, EDA, clustering, and the models)
- The cleaned/merged dataset as CSVs
- A short written report

## How the project works, roughly

1. Merged a bunch of different files from the dataset — activity, phone lock time, GPS, conversation detection, app usage, plus the self-report surveys and the actual academic records (grades, classes) — all joined by student and date so I ended up with one row per student per day.
2. Cleaned it up, handled missing values (a lot of columns are sparse since not everyone answered every survey every day), and did the usual EDA — distributions, outliers, correlations, that kind of thing.
3. Used clustering (K-Means and also Fuzzy C-Means) to group days into behavior patterns, based on a small set of behavior-related columns.
4. Added something I think is the more interesting part — instead of just comparing everyone to the average student, I built personalized baselines, so a student's behavior is compared to their own normal, not everyone else's.
5. Trained two models (Logistic Regression and Random Forest) to see how well they could pick up on this pattern, once using just the general features and once adding in the personalized ones.

## What I found

Adding the personalized baseline made a real difference — AUROC went from around 0.68–0.74 with just the general features to 0.92–0.94 once the personalized features were added. So basically, knowing how someone's behavior compares to their own normal matters a lot more than comparing them to everyone else.

The fuzzy clustering also picked up on a real pattern (not just noise), which was good to confirm since fuzzy methods can sometimes not really separate anything meaningfully.

## Limitations, being honest about them

- The deadline data I had access to only covered about 9 days out of the whole term, so the label isn't purely tied to real deadlines, it's based on behavior clustering.
- Only 49 students and one term of data, so I wouldn't say the exact numbers generalize, but the pattern (personalized features helping) is probably the more solid takeaway.
- Some of the personalized features are mathematically derived from the same raw data used to build the labels, so there's a bit of unavoidable overlap there. I tried to keep the labeling and modeling features separate as much as possible to avoid it being circular.
- Still need to double check which direction the grade scale goes (whether 1 means a good grade or a bad one) before I say for sure which behavior group is doing better or worse academically.

## Running it

Everything runs in Google Colab. First run, you'll need to upload the dataset zip (from Zenodo, StudentLife RDS version) — after that it's saved to Drive so you don't have to redo it every time. Just run the notebook top to bottom.

## Sources I used for the literature side

Mainly looked at the original StudentLife papers (Wang et al.), a couple of recent papers on procrastination prediction using ML (ACADPro, an autoencoder-based one, a decision tree one), and one closely related paper (ISense) that predicts self-regulated learning from phone sensing, which is the closest thing to what I'm doing but not quite the same target.
