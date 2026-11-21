"""
Unit Tests for Student Performance Prediction Pipeline (Standard Library unittest):
- Dataset loading and integrity tests
- Preprocessing and feature engineering checks
- Multi-stage model training and inference bounds
- SHAP local explainability validation
- What-if habit simulation checks
"""

import unittest
import pandas as pd
import numpy as np

from modules.member1_data_analysis import load_dataset
from modules.member2_ml_models import (
    preprocess_dataframe,
    load_and_train_pipeline,
    predict_student,
    simulate_study_time,
    simulate_absences
)


class TestStudentPipeline(unittest.TestCase):

    def test_data_loading(self):
        """Verifies that datasets load properly with target columns."""
        df_mat = load_dataset("Mathematics")
        self.assertEqual(len(df_mat), 395)
        self.assertIn("passed", df_mat.columns)
        self.assertIn("status", df_mat.columns)

        df_por = load_dataset("Portuguese Language")
        self.assertEqual(len(df_por), 649)

        df_comb = load_dataset("Combined")
        self.assertEqual(len(df_comb), 1044)

    def test_preprocessing(self):
        """Checks feature engineering logic."""
        sample_raw = pd.DataFrame([{
            "Medu": 4, "Fedu": 2,
            "Dalc": 1, "Walc": 3,
            "G1": 12, "G2": 15,
            "G3": 14,
            "schoolsup": "yes",
            "famsup": "no",
            "higher": "yes",
            "internet": "yes"
        }])
        processed = preprocess_dataframe(sample_raw)

        self.assertEqual(processed["parent_edu"].iloc[0], 3.0)
        self.assertEqual(processed["alcohol"].iloc[0], 2.0)
        self.assertEqual(processed["score_momentum"].iloc[0], 3)
        self.assertEqual(processed["passed"].iloc[0], 1)
        self.assertEqual(processed["schoolsup"].iloc[0], 1)
        self.assertEqual(processed["famsup"].iloc[0], 0)

    def test_multi_stage_models_and_metrics(self):
        """Checks that all 3 academic stages train and contain test metrics."""
        pipeline = load_and_train_pipeline()
        self.assertIn(1, pipeline)
        self.assertIn(2, pipeline)
        self.assertIn(3, pipeline)

        for stage_num in [1, 2, 3]:
            sinfo = pipeline[stage_num]
            metrics = sinfo["metrics"]
            self.assertTrue(0 <= metrics["accuracy"] <= 100)
            self.assertTrue(0 <= metrics["roc_auc"] <= 100)
            self.assertTrue(metrics["test_size"] > 0)
            self.assertTrue(metrics["train_size"] > 0)

    def test_prediction_output_bounds(self):
        """Checks that predictions produce valid bounded probabilities and SHAP outputs."""
        student_record = {
            "studytime": 3,
            "failures": 0,
            "absences": 4,
            "G1": 14,
            "G2": 15,
            "Medu": 3,
            "Fedu": 3,
            "schoolsup": 0,
            "famsup": 1,
            "higher": 1,
            "internet": 1,
            "goout": 2,
            "health": 4,
            "Dalc": 1.0,
            "Walc": 1.5
        }

        for stage in [1, 2, 3]:
            res = predict_student(student_record, stage=stage)
            self.assertIn(res["prediction"], [0, 1])
            self.assertTrue(0.0 <= res["pass_prob"] <= 100.0)
            self.assertTrue(0.0 <= res["fail_prob"] <= 100.0)
            self.assertEqual(round(res["pass_prob"] + res["fail_prob"], 1), 100.0)
            self.assertIn(res["status"], ["PASS", "FAIL / AT-RISK"])

            # Validate SHAP feature breakdown
            shap_df = res["shap_df"]
            self.assertIsInstance(shap_df, pd.DataFrame)
            self.assertTrue(len(shap_df) > 0)
            self.assertIn("Feature", shap_df.columns)
            self.assertIn("Impact", shap_df.columns)

    def test_simulations(self):
        """Verifies that sensitivity simulations return valid non-empty DataFrames."""
        student_record = {
            "studytime": 2,
            "failures": 1,
            "absences": 8,
            "G1": 10,
            "G2": 10
        }
        df_study = simulate_study_time(student_record, stage=3)
        self.assertEqual(len(df_study), 4)
        self.assertIn("Pass_Chance", df_study.columns)

        df_abs = simulate_absences(student_record, stage=3)
        self.assertTrue(len(df_abs) > 5)
        self.assertIn("Pass_Chance", df_abs.columns)


if __name__ == "__main__":
    unittest.main()
