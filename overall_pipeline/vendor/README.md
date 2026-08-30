# Vendored evaluation code

`spider_test_suite_eval/` is an unmodified checkout of the official Spider
test-suite SQL evaluator:

- upstream: https://github.com/taoyds/test-suite-sql-eval
- pinned commit: `e97acc546ecbee8fa27fa8dbf025ef61493a876c`
- license: Apache-2.0 (see the checkout's `LICENSE`)

Project code imports the evaluator only through `text2sql.official_eval` and
keeps its results separate from the project's local SQLite result comparison.
