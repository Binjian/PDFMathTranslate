import importlib
import importlib.util
import os
import sys
import tempfile
import unittest


class TestCliVersion(unittest.TestCase):
    def tearDown(self):
        for module_name in [
            "pdf2zh",
            "pdf2zh.pdf2zh",
            "pdf2zh.high_level",
            "pdf2zh.doclayout",
        ]:
            sys.modules.pop(module_name, None)

    def test_importing_package_does_not_eagerly_load_translation_pipeline(self):
        pkg = importlib.import_module("pdf2zh")

        self.assertEqual(pkg.__version__, "1.9.11")
        self.assertNotIn("pdf2zh.high_level", sys.modules)

    def test_importing_package_loads_dotenv_into_environment(self):
        if importlib.util.find_spec("dotenv") is None:
            self.skipTest("python-dotenv is not installed in this environment")

        original_cwd = os.getcwd()
        original_value = os.environ.pop("PDF2ZH_TEST_DOTENV", None)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.chdir(temp_dir)
                with open(".env", "w", encoding="utf-8") as handle:
                    handle.write("PDF2ZH_TEST_DOTENV=loaded-from-dotenv\n")

                importlib.import_module("pdf2zh")

                self.assertEqual(
                    os.environ.get("PDF2ZH_TEST_DOTENV"), "loaded-from-dotenv"
                )
        finally:
            os.chdir(original_cwd)
            if original_value is None:
                os.environ.pop("PDF2ZH_TEST_DOTENV", None)
            else:
                os.environ["PDF2ZH_TEST_DOTENV"] = original_value

    def test_importing_package_loads_repo_dotenv_when_cwd_has_no_dotenv(self):
        if importlib.util.find_spec("dotenv") is None:
            self.skipTest("python-dotenv is not installed in this environment")

        original_cwd = os.getcwd()
        original_value = os.environ.pop("DASHSCOPE_API_URL", None)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.chdir(temp_dir)

                importlib.import_module("pdf2zh")

                self.assertEqual(
                    os.environ.get("DASHSCOPE_API_URL"),
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
        finally:
            os.chdir(original_cwd)
            if original_value is None:
                os.environ.pop("DASHSCOPE_API_URL", None)
            else:
                os.environ["DASHSCOPE_API_URL"] = original_value

    def test_version_flag_exits_before_loading_heavy_modules(self):
        cli = importlib.import_module("pdf2zh.pdf2zh")

        self.assertNotIn("pdf2zh.high_level", sys.modules)
        self.assertNotIn("pdf2zh.doclayout", sys.modules)

        with self.assertRaises(SystemExit) as exit_context:
            cli.main(["-v"])

        self.assertEqual(exit_context.exception.code, 0)
        self.assertNotIn("pdf2zh.high_level", sys.modules)
        self.assertNotIn("pdf2zh.doclayout", sys.modules)


if __name__ == "__main__":
    unittest.main()
