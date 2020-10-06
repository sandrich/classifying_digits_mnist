"""Report Manager Tests"""
import json
import os
import random
import shutil
import string
import unittest
from mnist_classifier import report_manager

CORRECT_CONF_FILE = "tests/correct_conf.json"
NO_REPORT_DIR_CONF_FILE = "tests/no_report_conf.json"
NO_TEST_CONF_FILE = "tests/no_test_conf.json"


class ReportTest(unittest.TestCase):
    """Report Manager test class"""

    target_test_folder = ""

    def setUp(self):
        letters = string.ascii_lowercase
        self.target_test_folder = "".join(random.choice(letters) for i in range(10))
        json.dump({
            "report_directory": self.target_test_folder,
            "tests": [
                {
                    "type": "mlp",
                    "hidden_layers": "28 28 14",
                    "max_iter": 100,
                    "verbose": True,
                    "save": "mlp.model"
                }
            ]
        }, open(CORRECT_CONF_FILE, "w"))

        json.dump({
            "tests": [
                {
                    "type": "mlp",
                    "hidden_layers": "28 28 14",
                    "max_iter": 100,
                    "verbose": True,
                    "save": "mlp.model"
                }
            ]
        }, open(NO_REPORT_DIR_CONF_FILE, "w"))

        json.dump({
            "report_directory": "test_suite"
        }, open(NO_TEST_CONF_FILE, "w"))

    def tearDown(self):
        if not os.path.exists("reports"):
            return
        for test_folder in os.listdir("reports"):
            if self.target_test_folder in test_folder:
                shutil.rmtree(os.path.join("reports", test_folder))

        for path in [CORRECT_CONF_FILE, NO_TEST_CONF_FILE, NO_REPORT_DIR_CONF_FILE]:
            if os.path.exists(path):
                os.remove(path)

    def test_create_folder(self):
        """Test if a created folder exists and if the increment works"""
        with self.subTest():
            report_manager.prepare_report_dest(self.target_test_folder)
            self.assertTrue(os.path.exists(os.path.join("reports", self.target_test_folder)))
        with self.subTest():
            report_manager.prepare_report_dest(self.target_test_folder)
            self.assertTrue(os.path.exists(os.path.join("reports", self.target_test_folder + "_1")))
        with self.subTest():
            report_manager.prepare_report_dest(self.target_test_folder)
            self.assertTrue(os.path.exists(os.path.join("reports", self.target_test_folder + "_2")))

    def test_correct_folder(self):
        """Test the correct parsing of the json conf file"""
        out = report_manager.load_test_suite_conf(CORRECT_CONF_FILE)
        self.assertEqual(out, [['--report_directory', os.path.join("reports", self.target_test_folder),
                               '--save', 'mlp.model',
                               'mlp',
                               '--hidden_layers=28 28 14',
                               '--max_iter', '100',
                               '-v']])

    def test_error_conf_file(self):
        """Tests that the errors are properly raised"""
        with self.assertRaises(ValueError):
            _ = report_manager.load_test_suite_conf(NO_REPORT_DIR_CONF_FILE)

        with self.assertRaises(ValueError):
            _ = report_manager.load_test_suite_conf(NO_TEST_CONF_FILE)
