# SPDX-License-Identifier: Apache-2.0
"""The planted-defect registry: each entry is a defect and the one test that must catch it.

scripts/check_planted_defects.py plants each defect in a fresh clone of HEAD, runs the named test, and fails unless
that test passes before the defect is planted and fails after. tests/test_planted_defects.py checks that every entry
still applies: its text occurs exactly once in its file, and its test is collected. Entries marked "gpu" need a
CUDA device and run on the Jetson only.
"""

DEFECTS = [
    {
        "id": "readme-tree-stale",
        "file": "README.md",
        "old": "├── LICENSE\n",
        "new": "",
        "catches": "tests/test_readme_tree.py::test_readme_tree_equals_generated_tree",
    },
    {
        "id": "agents-rule-edited",
        "file": "AGENTS.md",
        "old": "Subject: imperative, no trailing period",
        "new": "Subject: imperative, no trailing full stop",
        "catches": "tests/test_agents_rule.py::test_agents_md_carries_the_commit_rule_verbatim",
    },
    {
        "id": "test-name-shared",
        "file": "tests/test_run_logs.py",
        "old": "def test_wrong_type_is_rejected():",
        "new": "def test_valid_log_passes():  # noqa: F811",
        "catches": "tests/test_test_names.py::test_no_two_tests_share_a_name",
    },
    {
        "id": "ci-command-changed",
        "file": ".github/workflows/ci.yml",
        "old": "python -m pytest -rs",
        "new": "python -m pytest",
        "catches": "tests/test_ci_contract.py::test_workflow_runs_what_the_readme_says",
    },
    {
        "id": "ci-python-dropped",
        "file": ".github/workflows/ci.yml",
        "old": '"3.10", ',
        "new": "",
        "catches": "tests/test_ci_contract.py::test_workflow_platform_is_what_the_readme_says",
    },
    {
        "id": "ci-strict-skips-dropped",
        "file": ".github/workflows/ci.yml",
        "old": "PDORIN_STRICT_SKIPS=1 python -m pytest",
        "new": "python -m pytest",
        "catches": "tests/test_ci_contract.py::test_ci_makes_an_unexpected_skip_fail_the_run",
    },
    {
        "id": "ci-action-by-tag",
        "file": ".github/workflows/ci.yml",
        "old": "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0",
        "new": "actions/setup-python@v5",
        "catches": "tests/test_ci_contract.py::test_every_action_is_pinned_by_sha",
    },
    {
        "id": "ci-local-drift",
        "file": "scripts/ci_local.sh",
        "old": 'VERSIONS="3.10 3.11 3.12"',
        "new": 'VERSIONS="3.12"',
        "catches": "tests/test_ci_contract.py::test_local_script_mirrors_the_ci_matrix",
    },
    {
        "id": "version-disagrees",
        "file": "CITATION.cff",
        "old": "version: 0.1.0",
        "new": "version: 0.2.0",
        "catches": "tests/test_version_agreement.py::test_versions_agree",
    },
    {
        "id": "changelog-entry-missing",
        "file": "CHANGELOG.md",
        "old": "## [0.1.0] - ",
        "new": "## [0.0.9] - ",
        "catches": "tests/test_changelog.py::test_every_version_has_a_changelog_entry",
    },
    {
        "id": "doc-cites-missing-test",
        "file": "CONTRIBUTING.md",
        "old": "`tests/test_changelog.py`",
        "new": "`tests/test_changelogs.py`",
        "catches": "tests/test_doc_test_refs.py::test_every_cited_test_is_collected",
    },
    {
        "id": "prd-requirement-uncovered",
        "file": "PRD.md",
        "old": "`tests/test_postprocess.py::test_pipeline_equals_torchvision_detector`",
        "new": "none",
        "catches": "tests/test_prd_coverage.py::test_every_requirement_is_covered_by_a_ci_test",
    },
    {
        "id": "commit-id-not-in-history",
        "file": "README.md",
        "old": "\n## Limitations\n",
        "new": "\nMeasured at commit 0123abcd.\n\n## Limitations\n",
        "catches": "tests/test_commit_ids.py::test_every_cited_commit_is_in_the_history",
    },
    {
        "id": "int8-limit-raised",
        "file": "scripts/generate_table.py",
        "old": "INT8_DROP_LIMIT_POINTS = 2.0",
        "new": "INT8_DROP_LIMIT_POINTS = 3.0",
        "catches": "tests/test_requirements.py::test_r10_an_int8_drop_over_two_points_is_stated",
    },
    {
        "id": "nms-ignores-class",
        "file": "src/peddet_orin/postprocess.py",
        "old": "batched_nms(all_boxes, all_scores, all_labels, ",
        "new": "batched_nms(all_boxes, all_scores, all_labels * 0, ",
        "catches": "tests/test_postprocess.py::test_overlapping_boxes_of_different_classes_both_survive",
    },
    {
        "id": "runlog-fields-unchecked",
        "file": "src/peddet_orin/runlog.py",
        "old": "for field, expected in REQUIRED_FIELDS.items():",
        "new": "for field, expected in list(REQUIRED_FIELDS.items())[:0]:",
        "catches": "tests/test_run_logs.py::test_wrong_type_is_rejected",
    },
    {
        "id": "cuda-forward-other-weights",
        "file": "tests/test_gpu.py",
        "old": 'PytorchBackend(detector, "cuda")(batch)',
        "new": 'PytorchBackend(lively_detector(SEED + 1), "cuda")(batch)',
        "catches": "tests/test_gpu.py::test_cuda_forward_matches_cpu",
        "needs": "gpu",
    },
]
