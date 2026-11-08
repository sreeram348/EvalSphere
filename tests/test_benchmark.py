import json
import unittest
from io import BytesIO

import numpy as np
import pandas as pd

from benchmark import (
    MODEL_NAMES,
    compare_classifiers,
    compare_clusters,
    compare_regressors,
    dataset_sources,
    read_dataset,
)


class DatasetReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {"feature": [1, 2, 3], "category": ["a", "b", "a"]}
        )

    def test_reads_csv_and_json(self) -> None:
        csv_data = self.data.to_csv(index=False).encode()
        json_text = self.data.to_json(orient="records")
        if json_text is None:
            raise AssertionError("Expected JSON text from the in-memory dataframe.")
        json_data = json_text.encode()

        pd.testing.assert_frame_equal(
            read_dataset("sample.csv", csv_data),
            self.data,
        )
        pd.testing.assert_frame_equal(
            read_dataset("sample.json", json_data),
            self.data,
        )

    def test_reads_markdown_pipe_table(self) -> None:
        markdown = b"| feature | category |\n| --- | :---: |\n| 1 | a |\n| 2 | b |\n"
        result = read_dataset("sample.md", markdown)

        self.assertEqual(list(result.columns), ["feature", "category"])
        self.assertEqual(result["feature"].tolist(), ["1", "2"])

    def test_reads_html_tables_and_lists_sources(self) -> None:
        html = (
            b"<table><tr><th>feature</th></tr><tr><td>1</td></tr></table>"
            b"<table><tr><th>feature</th></tr><tr><td>2</td></tr></table>"
        )

        self.assertEqual(
            dataset_sources("sample.html", html), ["Table 1", "Table 2"]
        )
        self.assertEqual(
            read_dataset("sample.html", html, source="Table 2")["feature"].tolist(),
            [2],
        )
        with self.assertRaisesRegex(ValueError, "Selected HTML table"):
            read_dataset("sample.html", html, source="Table 3")

    def test_reads_xml_table(self) -> None:
        xml = (
            b"<root><row><feature>1</feature><category>a</category></row>"
            b"<row><feature>2</feature><category>b</category></row></root>"
        )
        result = read_dataset("sample.xml", xml)

        self.assertEqual(result["feature"].tolist(), [1, 2])
        self.assertEqual(result["category"].tolist(), ["a", "b"])

    def test_reads_excel_sheets_and_parquet(self) -> None:
        excel = BytesIO()
        with pd.ExcelWriter(excel, engine="openpyxl") as writer:
            self.data.to_excel(writer, sheet_name="Training", index=False)
            self.data.iloc[:2].to_excel(writer, sheet_name="Validation", index=False)
        excel_bytes = excel.getvalue()

        self.assertEqual(
            dataset_sources("sample.xlsx", excel_bytes), ["Training", "Validation"]
        )
        pd.testing.assert_frame_equal(
            read_dataset("sample.xlsx", excel_bytes, source="Training"),
            self.data,
        )

        parquet = BytesIO()
        self.data.to_parquet(parquet, index=False)
        pd.testing.assert_frame_equal(
            read_dataset("sample.parquet", parquet.getvalue()),
            self.data,
        )

    def test_rejects_unsupported_extensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported file extension"):
            read_dataset("sample.pdf", b"not a dataset")

    def test_rejects_malformed_tabular_files_as_input_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not inspect the Excel workbook"):
            read_dataset("broken.xlsx", b"not an Excel workbook")
        with self.assertRaisesRegex(ValueError, "Could not read"):
            read_dataset("broken.xml", b"<root>")
        with self.assertRaisesRegex(ValueError, "Could not read"):
            read_dataset("broken.parquet", b"not a parquet file")


class AutoMLBenchmarkTests(unittest.TestCase):
    def test_compares_classification_models_and_ranks_results(self) -> None:
        rng = np.random.default_rng(42)
        numeric = rng.normal(size=60)
        dataset = pd.DataFrame(
            {
                "signal": numeric,
                "noise": rng.normal(size=60),
                "group": np.where(numeric > 0, "positive", "negative"),
                "target": np.where(numeric > 0, "yes", "no"),
            }
        )

        results, predictions, holdout, class_labels, relevance = compare_classifiers(
            dataset, "target"
        )

        self.assertEqual(len(results), len(MODEL_NAMES))
        self.assertEqual(results["Rank"].tolist(), list(range(1, len(MODEL_NAMES) + 1)))
        self.assertIn("Random Forest", predictions)
        self.assertEqual(len(predictions["Random Forest"]), len(holdout))
        self.assertEqual(class_labels, ["no", "yes"])
        self.assertIn("signal", relevance["Feature"].tolist())

    def test_compares_regression_models(self) -> None:
        rng = np.random.default_rng(11)
        feature = rng.normal(size=40)
        dataset = pd.DataFrame(
            {
                "feature": feature,
                "category": np.where(feature > 0, "high", "low"),
                "target": 2 * feature + rng.normal(scale=0.1, size=40),
            }
        )

        results, predictions, holdout = compare_regressors(dataset, "target")

        self.assertEqual(results["Rank"].tolist(), list(range(1, len(results) + 1)))
        self.assertIn("R2", results.columns)
        self.assertEqual(len(predictions[results.iloc[0]["Model"]]), len(holdout))

    def test_compares_unsupervised_clusterers(self) -> None:
        rng = np.random.default_rng(5)
        dataset = pd.DataFrame(
            {
                "x": np.concatenate(
                    [rng.normal(-3, 0.2, 20), rng.normal(3, 0.2, 20)]
                ),
                "y": np.concatenate(
                    [rng.normal(-3, 0.2, 20), rng.normal(3, 0.2, 20)]
                ),
                "known_group": ["A"] * 20 + ["B"] * 20,
            }
        )

        results, labels, features = compare_clusters(
            dataset, excluded_columns=("known_group",), n_clusters=2
        )

        self.assertEqual(results["Rank"].tolist(), [1, 2, 3])
        self.assertIn("Silhouette", results.columns)
        self.assertTrue(np.isfinite(results["Silhouette"]).all())
        self.assertNotIn("known_group", features.columns)
        self.assertEqual(len(labels["K-Means"]), len(dataset))

    def test_tunes_leading_classification_and_regression_models(self) -> None:
        rng = np.random.default_rng(42)
        feature = rng.normal(size=60)
        classification_data = pd.DataFrame(
            {
                "signal": feature,
                "noise": rng.normal(size=60),
                "target": np.where(feature > 0, "yes", "no"),
            }
        )
        classification_results = compare_classifiers(
            classification_data,
            "target",
            tune_best=True,
        )[0]
        self.assertTrue(classification_results["Tuned parameters"].ne("").any())

        rng = np.random.default_rng(11)
        feature = rng.normal(size=40)
        regression_data = pd.DataFrame(
            {
                "feature": feature,
                "target": 2 * feature + rng.normal(scale=0.1, size=40),
            }
        )
        regression_results = compare_regressors(
            regression_data,
            "target",
            tune_best=True,
        )[0]
        self.assertTrue(regression_results["Tuned parameters"].ne("").any())


if __name__ == "__main__":
    unittest.main()
