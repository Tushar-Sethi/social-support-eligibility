import pandas as pd
import numpy as np

def generate_model_explanation_prompt(pipeline, feature_df, prediction):
    """
    Build a prompt describing why the RandomForest pipeline predicted 'prediction'
    for the single row in feature_df. Returns a string you can send to your LLM.

    Assumes:
      - 'pipeline' is a sklearn Pipeline whose last step is a RandomForestClassifier,
        and whose first step is a ColumnTransformer preprocessing numeric + categorical features.
      - 'feature_df' is a one‐row pandas DataFrame containing exactly the columns the model expects.
      - 'prediction' is the model output for that row (0 or 1).

    Steps:
      1. Use pipeline.named_steps['preprocessor'] to transform the row into a feature vector.
      2. Extract feature names via preprocessor.get_feature_names_out().
      3. Grab classifier = pipeline.named_steps['classifier'] and its feature_importances_.
      4. Pair names + importances, sort descending, keep top 3.
      5. For each top feature, read off the actual value from the transformed vector.
      6. Compose a prompt like:

         “You are an AI assistant. The model predicted <prediction> (meaning ‘eligible’ or ‘not eligible’).
          Here are the three factors that mattered most, with their importances and the applicant’s values:
           1. <feature1_name> (importance = 0.24): user’s value = <…>
           2. <feature2_name> (importance = 0.15): user’s value = <…>
           3. <feature3_name> (importance = 0.10): user’s value = <…>

          In plain language, please explain why those factors (and the user’s values for them) led to <prediction>.”

    Returns:
      A single string (the prompt).
    """
    # 1) Extract preprocessor and classifier
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    # 2) Transform the single‐row DataFrame into the model’s internal 2D array
    X_transformed = preprocessor.transform(feature_df)  # shape (1, n_features_total)
    # Flatten to 1D
    x_row = X_transformed[0]

    # 3) Get feature names after column‐transformer
    #    (sklearn ≥1.0: use get_feature_names_out())
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        # Fallback (older sklearn)
        # numeric features followed by one-hot columns; you may need to adjust this if your version differs
        num_feats = preprocessor.transformers_[0][2]
        cat_feats = preprocessor.transformers_[1][2]
        # OneHotEncoder has a get_feature_names_out on its own
        ohe = preprocessor.transformers_[1][1].named_steps['onehot']
        cat_ohe_names = ohe.get_feature_names_out(cat_feats)
        feature_names = np.concatenate([num_feats, cat_ohe_names])

    # 4) Get importances array from classifier
    importances = classifier.feature_importances_

    # 5) Pair names & importances, sort by importance descending
    idx_sorted = np.argsort(importances)[::-1]
    top_n = 3  # you can change to top 5 if desired
    top_idx = idx_sorted[:top_n]

    lines = []
    for rank, idx in enumerate(top_idx, start=1):
        fname = feature_names[idx]
        imp = importances[idx]
        # Corresponding value in x_row
        val = x_row[idx]
        # If it’s a one-hot categorical column, val is either 0.0 or 1.0. Convert to “Yes/No” or category name:
        if val in [0.0, 1.0] and fname.split("__")[0].startswith("cat"):
            # format "cat__Gender_Male" → "Gender = Male"
            parts = fname.split("__")[1]  # e.g. "Gender_Male"
            feat_name, feat_val = parts.split("_", 1)
            human_val = f"{feat_val} (Yes)"
        else:
            # Numeric
            human_val = f"{val:.2f}"
            feat_name = fname.split("__")[-1] if "__" in fname else fname
        lines.append(
            f"{rank}. **{feat_name}** (importance={imp:.3f}) — user’s value: {human_val}"
        )

    # 6) Decide a human label for prediction
    label_text = "eligible for support" if prediction == 1 else "not eligible for support"

    # 7) Build the final prompt
    explanation = (
        f"You are an AI assistant that explains a classification model’s decision to an end user.\n"
        f"The model predicted **{label_text}** (label={prediction}).\n\n"
        f"Below are the three features that influenced this decision the most, "
        f"along with the model’s importance weight and the applicant’s actual value:\n\n"
        + "\n".join(lines)
        + "\n\nIn plain language, please explain why these factors and values led to **"
        + label_text
        + "**."
        + "IMPORTANT NOTES: If the applicant is ineligible, please explain why they are ineligible in a very polite and professional manner."
        + "Please be very specific and detailed in your explanation."
        + "Please be very polite and professional in your explanation."
        + "If the applicant is eligible, please explain why they are eligible in a very polite and professional manner."
        + "Please be mindful of the fact that whatever you generate will be directly sent to the applicant."
    )
    return explanation
