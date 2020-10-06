import unittest
from src import report_manager
import random
import string
import shutil
import os


class ReportTest(unittest.TestCase):

    target_test_folder = ""

    @classmethod
    def setUpClass(self):
        letters = string.ascii_lowercase
        self.target_test_folder = "".join(random.choice(letters) for i in range(10))

    @classmethod
    def tearDownClass(self):
        if not os.path.exists("reports"):
            return
        for test_folder in os.listdir("reports"):
            if self.target_test_folder in test_folder:
                shutil.rmtree(os.path.join("reports", test_folder))

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
