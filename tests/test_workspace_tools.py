"""Workspace sandbox, file tools and propose -> apply -> rollback. Uses a temp
git repo seeded from examples/workspace; no server needed."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from onnx_lfm_agent import workspace_tools as wt
from onnx_lfm_agent.agent import Agent
from onnx_lfm_agent.toolsets import build_registry
from onnx_lfm_agent.tools import Registry
from onnx_lfm_agent.workspace import Workspace, WorkspaceError

from fakes import FakeClient, reply, tool_call

SEED = Path(__file__).parent.parent / "examples" / "workspace"


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=Human",
                           "-c", "user.email=human@example.com", *args],
                          check=True, capture_output=True, text=True).stdout


@pytest.fixture
def root(tmp_path):
    ws = tmp_path / "ws"
    shutil.copytree(SEED, ws)
    git(ws, "init", "-q", "-b", "main")
    git(ws, "add", "-A")
    git(ws, "commit", "-q", "-m", "seed")
    return ws


@pytest.fixture
def tools(root):
    r = Registry()
    wt.register(r, Workspace(root))
    return {t.name: t for t in r}


def call(tools, name, **kw):
    return tools[name].func(**kw)


# --- sandbox -----------------------------------------------------------------

@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "logs/../../x", ".git/config"])
def test_paths_outside_or_into_git_refused(tools, path):
    with pytest.raises(WorkspaceError):
        call(tools, "read_file", path=path)


def test_symlink_out_of_workspace_refused(tools, root, tmp_path):
    (tmp_path / "secret.txt").write_text("nope")
    (root / "link.txt").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        call(tools, "read_file", path="link.txt")
    assert "link.txt" not in call(tools, "list_files")["entries"]


def test_absolute_path_inside_workspace_ok(tools, root):
    assert call(tools, "read_file", path=str(root / "app.yaml"))["path"] == "app.yaml"


def test_missing_workspace_is_a_clear_error(tmp_path):
    r = Registry()
    wt.register(r, Workspace(tmp_path / "nope"))
    with pytest.raises(WorkspaceError, match="workspace directory not found"):
        r.get("list_files").func()


# --- read-only tools ---------------------------------------------------------

def test_list_files_hides_git(tools):
    out = call(tools, "list_files")
    assert out == {"entries": ["logs/", "app.schema.json", "app.yaml", "logs/app.log"],
                   "truncated": False}


def test_list_files_caps(tools, root, monkeypatch):
    monkeypatch.setattr(wt, "MAX_LIST", 2)
    out = call(tools, "list_files")
    assert len(out["entries"]) == 2 and out["truncated"]


def test_read_file_numbers_and_ranges(tools):
    out = call(tools, "read_file", path="app.yaml", start_line=3, max_lines=2)
    assert out["content"] == "3:   host: 0.0.0.0\n4:   port: 8080"
    assert out["truncated"] and out["total_lines"] == 12


def test_read_file_char_cap(tools, root, monkeypatch):
    monkeypatch.setattr(wt, "MAX_READ_CHARS", 30)
    out = call(tools, "read_file", path="logs/app.log")
    assert out["content"] == ""  # first line alone is over the cap
    assert out["truncated"]


def test_read_file_refuses_binary(tools, root):
    (root / "blob.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(WorkspaceError, match="binary"):
        call(tools, "read_file", path="blob.bin")


def test_search_files(tools):
    out = call(tools, "search_files", text="TIMED OUT")
    assert len(out["matches"]) == 3
    assert out["matches"][0].startswith("logs/app.log:3: 2026-10-02 14:05:40 ERROR")


def test_search_files_caps(tools, monkeypatch):
    monkeypatch.setattr(wt, "MAX_MATCHES", 1)
    out = call(tools, "search_files", text="GET", path="logs")
    assert len(out["matches"]) == 1 and out["truncated"]


# --- propose -----------------------------------------------------------------

def test_propose_writes_nothing_and_keeps_yaml_comments(tools, root):
    before = (root / "app.yaml").read_text()
    out = call(tools, "propose_config_change", path="app.yaml",
               key="server.request_timeout_s", value=15, reason="billing p99 is 11.8s")
    assert out["valid"] and out["proposal_id"] == "p1"
    assert out["schema"] == "app.schema.json"
    assert "-  request_timeout_s: 5   # upstream calls to the billing API" in out["diff"]
    # The comment keeps its column, so the longer value eats one space.
    assert "+  request_timeout_s: 15  # upstream calls to the billing API" in out["diff"]
    assert out["diff"].count("\n+") == 2  # header + exactly one changed line
    assert (root / "app.yaml").read_text() == before


def test_propose_schema_violation_has_no_proposal_id(tools):
    out = call(tools, "propose_config_change", path="app.yaml", key="server.port", value="eighty")
    assert not out["valid"] and out["proposal_id"] is None
    assert out["errors"] == ["server.port: 'eighty' is not of type 'integer'"]
    assert "note" not in out


# --- narrow repair of quoted numbers/booleans ---------------------------------

def test_quoted_number_converted_when_schema_wants_number(tools):
    out = call(tools, "propose_config_change", path="app.yaml",
               key="server.request_timeout_s", value="15")
    assert out["valid"] and out["proposal_id"]
    assert "+  request_timeout_s: 15  #" in out["diff"]  # unquoted in the file
    assert out["note"] == "converted '15' (string) to 15 to match app.schema.json"


def test_no_conversion_when_schema_accepts_the_string(tools):
    out = call(tools, "propose_config_change", path="app.yaml", key="server.host", value="8080")
    assert out["valid"] and "note" not in out
    assert "+  host: '8080'" in out["diff"]


def test_no_conversion_when_converted_value_also_fails(tools):
    out = call(tools, "propose_config_change", path="app.yaml", key="server.port", value="99999")
    assert not out["valid"] and "note" not in out
    assert out["errors"] == ["server.port: '99999' is not of type 'integer'"]


@pytest.mark.parametrize("current, sent, converted", [
    ("8", "16", 16), ("1.5", "2.5", 2.5), ("true", "False", False),
    ("label", "16", None),     # current is a string: keep the string
    ("8", "true", None),       # number vs boolean: different kinds
])
def test_conversion_without_schema_follows_current_type(tools, root, current, sent, converted):
    (root / "plain.yaml").write_text(f"x: {current}\n")
    git(root, "add", "plain.yaml")
    git(root, "commit", "-q", "-m", "plain")
    out = call(tools, "propose_config_change", path="plain.yaml", key="x", value=sent)
    if converted is None:
        assert "note" not in out
    else:
        assert out["note"].endswith("to match the current value's type")
        assert f"+x: {str(converted).lower()}" in out["diff"]


def test_propose_without_schema_only_checks_parse(tools, root):
    (root / "extra.json").write_text('{\n    "a": {"b": 1}\n}\n')
    git(root, "add", "extra.json")
    git(root, "commit", "-q", "-m", "extra")
    out = call(tools, "propose_config_change", path="extra.json", key="a.c", value=[1, 2])
    assert out["valid"] and out["proposal_id"]
    assert out["schema"].startswith("none found")
    assert '+        "c": [' in out["diff"]  # 4-space indent preserved


def test_propose_delete_and_list_index(tools, root):
    (root / "extra.yaml").write_text("workers:\n  - name: a\n  - name: b\n")
    git(root, "add", "extra.yaml")
    git(root, "commit", "-q", "-m", "extra")
    out = call(tools, "propose_config_change", path="extra.yaml", key="workers.1.name", value="c")
    assert "+  - name: c" in out["diff"]
    assert out["diff"].count("\n-") == 1  # one changed line: indentation kept
    out = call(tools, "propose_config_change", path="extra.yaml", key="workers.0", delete=True)
    assert "-  - name: a" in out["diff"]


def test_yaml_four_space_mapping_indent_kept(tools, root):
    (root / "four.yaml").write_text("a:\n    b: 1\n    c:\n    - x\n")
    git(root, "add", "four.yaml")
    git(root, "commit", "-q", "-m", "four")
    out = call(tools, "propose_config_change", path="four.yaml", key="a.b", value=2)
    assert out["diff"].endswith(" a:\n-    b: 1\n+    b: 2\n     c:\n     - x\n")


@pytest.mark.parametrize("kw, match", [
    ({"key": "server.nope.x", "value": 1},
     r"key not found: server.nope \(keys under server: host, port, request_timeout_s\)"),
    ({"key": "server..port", "value": 1}, "bad key"),
    ({"key": "server.port"}, "either a value or delete"),
    ({"key": "server.port", "value": 1, "delete": True}, "either a value or delete"),
    ({"key": "server.nope", "delete": True}, r"key not found: server.nope \(keys under server"),
    ({"key": "nope.x", "value": 1}, r"\(keys under the top level: server, database, logging\)"),
])
def test_propose_bad_requests(tools, kw, match):
    with pytest.raises(Exception, match=match):
        call(tools, "propose_config_change", path="app.yaml", **kw)


def test_new_key_rejected_by_schema_gets_hint(tools):
    out = call(tools, "propose_config_change", path="app.yaml", key="server.timeout", value=30)
    assert not out["valid"] and out["proposal_id"] is None
    assert out["hint"] == ("server.timeout is a new key; "
                           "keys under server: host, port, request_timeout_s")


def test_propose_refuses_non_config_and_dirty_files(tools, root):
    with pytest.raises(Exception, match="not a config file"):
        call(tools, "propose_config_change", path="logs/app.log", key="a", value=1)
    (root / "app.yaml").write_text((root / "app.yaml").read_text() + "# human edit\n")
    with pytest.raises(WorkspaceError, match="uncommitted changes"):
        call(tools, "propose_config_change", path="app.yaml", key="server.port", value=81)


def test_propose_needs_git_repo(tmp_path):
    shutil.copytree(SEED, tmp_path / "plain")
    r = Registry()
    wt.register(r, Workspace(tmp_path / "plain"))
    with pytest.raises(WorkspaceError, match="not the root of a git repository"):
        r.get("propose_config_change").func(path="app.yaml", key="server.port", value=81)


# --- narrow repair: inferring a missing path -----------------------------------

def test_path_taken_from_key(tools):
    out = call(tools, "propose_config_change", key="app.yaml.logging.level", value="debug")
    assert out["valid"] and out["path"] == "app.yaml"
    assert "+  level: debug" in out["diff"]
    assert out["note"] == "path not given; took app.yaml from key 'app.yaml.logging.level'"


def test_only_config_file_used_when_path_missing(tools):
    # app.schema.json doesn't count: schemas aren't editable configs here.
    out = call(tools, "propose_config_change", key="server.request_timeout_s", value="15")
    assert out["valid"] and out["path"] == "app.yaml"
    assert out["note"] == ("path not given; used the only config file, app.yaml; "
                           "converted '15' (string) to 15 to match app.schema.json")


def test_file_name_stripped_when_path_also_given(tools):
    out = call(tools, "propose_config_change", path="app.yaml",
               key="app.yaml.server.port", value=81)
    assert out["valid"] and "+  port: 81" in out["diff"]
    assert out["note"] == "removed the file name from key 'app.yaml.server.port'"


def test_longest_matching_file_wins(tools, root):
    (root / "conf").mkdir()
    (root / "conf" / "app.yaml").write_text("server:\n  port: 1\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "second app.yaml")
    out = call(tools, "propose_config_change", key="conf/app.yaml.server.port", value=2)
    assert out["path"] == "conf/app.yaml"


def test_ambiguous_missing_path_is_refused(tools, root):
    (root / "other.json").write_text('{"a": 1}\n')
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "other")
    with pytest.raises(WorkspaceError, match="path is required.*candidates: app.yaml, other.json"):
        call(tools, "propose_config_change", key="server.port", value=81)


# --- apply / rollback --------------------------------------------------------

def test_apply_commits_only_that_file_with_trailer(tools, root):
    (root / "notes.txt").write_text("human, untracked\n")
    pid = call(tools, "propose_config_change", path="app.yaml", key="server.request_timeout_s",
               value=15, reason="billing p99 is 11.8s")["proposal_id"]
    out = call(tools, "apply_config_change", proposal_id=pid)
    assert out["applied"] == pid and out["path"] == "app.yaml"

    assert "request_timeout_s: 15  # upstream" in (root / "app.yaml").read_text()
    log = git(root, "log", "-1", "--format=%an|%s|%b")
    assert log.startswith("onnx-lfm-agent|agent: set server.request_timeout_s in app.yaml|")
    assert "billing p99 is 11.8s" in log and "Agent-Proposal: p1" in log
    assert git(root, "show", "--name-only", "--format=", "HEAD").split() == ["app.yaml"]
    assert git(root, "status", "--porcelain").strip() == "?? notes.txt"


def test_apply_is_single_use(tools):
    pid = call(tools, "propose_config_change", path="app.yaml", key="server.port", value=81)["proposal_id"]
    call(tools, "apply_config_change", proposal_id=pid)
    with pytest.raises(WorkspaceError, match="unknown or already used"):
        call(tools, "apply_config_change", proposal_id=pid)


def test_apply_refuses_stale_proposal(tools, root):
    pid = call(tools, "propose_config_change", path="app.yaml", key="server.port", value=81)["proposal_id"]
    (root / "app.yaml").write_text((root / "app.yaml").read_text().replace("info", "debug"))
    with pytest.raises(WorkspaceError, match="changed since p1 was proposed"):
        call(tools, "apply_config_change", proposal_id=pid)


def test_apply_restores_file_if_commit_fails(tools, root):
    pid = call(tools, "propose_config_change", path="app.yaml", key="server.port", value=81)["proposal_id"]
    before = (root / "app.yaml").read_text()
    hook = root / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    with pytest.raises(WorkspaceError, match="git commit failed"):
        call(tools, "apply_config_change", proposal_id=pid)
    assert (root / "app.yaml").read_text() == before
    assert git(root, "status", "--porcelain").strip() == ""


def test_rollback_is_not_a_model_tool(tools):
    assert "rollback_config" not in tools


def test_rollback_reverts_agent_commit_only_once(tools, root):
    original = (root / "app.yaml").read_text()
    pid = call(tools, "propose_config_change", path="app.yaml", key="server.port", value=81)["proposal_id"]
    call(tools, "apply_config_change", proposal_id=pid)
    out = wt.rollback_last_change(Workspace(root))
    assert out["subject"] == "agent: set server.port in app.yaml"
    assert (root / "app.yaml").read_text() == original
    with pytest.raises(WorkspaceError, match="was not made by apply_config_change"):
        wt.rollback_last_change(Workspace(root))  # HEAD is now the revert


def test_rollback_refuses_human_commit(root):
    with pytest.raises(WorkspaceError, match="'seed'.*refusing"):
        wt.rollback_last_change(Workspace(root))


@pytest.fixture
def cli_ws(root, monkeypatch):
    from onnx_lfm_agent import cli
    monkeypatch.setattr(cli.settings, "workspace", str(root))
    return cli


def test_cli_rollback_confirms_then_reverts(cli_ws, tools, root, monkeypatch, capsys):
    pid = call(tools, "propose_config_change", path="app.yaml", key="server.port", value=81)["proposal_id"]
    call(tools, "apply_config_change", proposal_id=pid)
    monkeypatch.setattr("builtins.input", lambda q: "n")
    assert cli_ws._rollback(assume_yes=False) == 1
    assert "port: 81" in (root / "app.yaml").read_text()
    monkeypatch.setattr("builtins.input", lambda q: "y")
    assert cli_ws._rollback(assume_yes=False) == 0
    assert "port: 8080" in (root / "app.yaml").read_text()
    assert "will revert" in capsys.readouterr().out


def test_cli_rollback_refuses_human_commit(cli_ws, capsys):
    assert cli_ws._rollback(assume_yes=True) == 1
    assert "not made by the agent" in capsys.readouterr().err


def test_apply_preview_shows_diff(tools):
    pid = call(tools, "propose_config_change", path="app.yaml", key="server.port", value=81)["proposal_id"]
    text = tools["apply_config_change"].preview({"proposal_id": pid})
    assert text.startswith("agent: set server.port in app.yaml\n")
    assert "+  port: 81" in text


# --- policies, toolsets, full loop -------------------------------------------

def test_default_policies(root):
    agent = Agent(build_registry("workspace", workspace=str(root)), client=object(), tool_policy="")
    assert agent.policies == {
        "list_files": "allow", "read_file": "allow", "search_files": "allow",
        "propose_config_change": "allow", "apply_config_change": "ask"}


def test_toolsets(root):
    assert len(build_registry("workspace,demo", workspace=str(root))) == 7
    assert [t.name for t in build_registry("demo")] == ["get_current_time", "add"]
    with pytest.raises(ValueError, match="unknown toolset"):
        build_registry("workspace,shell")


def test_log_to_config_loop(root):
    """Model reads the log, proposes, applies (approved), then answers."""
    client = FakeClient([
        reply(None, tool_call(json.dumps({"text": "timed out"}), name="search_files", id="s")),
        reply(None, tool_call(json.dumps({"path": "app.yaml", "key": "server.request_timeout_s",
                                          "value": 15, "reason": "timeouts at 5s"}),
                              name="propose_config_change", id="p")),
        reply(None, tool_call('{"proposal_id": "p1"}', name="apply_config_change", id="a")),
        reply("Raised request_timeout_s to 15."),
    ])
    approved = []
    agent = Agent(build_registry("workspace", workspace=str(root)), client=client,
                  approve=lambda t, a: approved.append(t.preview(a)) or True, tool_policy="")
    result = agent.run("fix the timeouts in the logs")

    assert [s.status for s in result.steps] == ["executed", "executed", "executed"]
    assert "+  request_timeout_s: 15" in approved[0]
    assert "request_timeout_s: 15" in (root / "app.yaml").read_text()
    assert "Agent-Proposal: p1" in git(root, "log", "-1", "--format=%b")
