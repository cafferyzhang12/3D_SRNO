"""Exercise administrative submitters with a local controller/accounting simulator."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture()
def scheduler(tmp_path):
    bash = shutil.which("bash")
    if bash is None and Path("C:/Program Files/Git/bin/bash.exe").is_file():
        bash = "C:/Program Files/Git/bin/bash.exe"
    if bash is None:
        pytest.skip("Bash unavailable")
    scripts = tmp_path/"scripts"
    scripts.mkdir()
    source = Path(__file__).resolve().parents[1]/"scripts"
    for p in ("submit_mri_haar.sh","submit_mri_haar_study.sh","submit_mri_confirmation.sh","mri_slurm_dependencies.sh"):
        shutil.copyfile(source/p, scripts/p)
        subprocess.run([bash,"-n",f"scripts/{p}"], cwd=tmp_path,check=True)
    db = tmp_path/"db"
    db.mkdir()
    binaries = tmp_path/"mockbin"
    binaries.mkdir()
    for name, body in {
        "flock": "exit 0",
        "squeue": """
while [[ $1 != -j ]]; do shift; done
job=$2
if [[ -f "db/$job.delay" ]]; then rm "db/$job.delay"; exit 0; fi
cat "db/$job.queue" 2>/dev/null
""",
        "sacct": """
while [[ $1 != -j ]]; do shift; done
cat "db/$2.acct" 2>/dev/null
""",
        "sbatch": """
echo "$*" >> db/submitted
echo 'sbatch: warning: mock partition message' >&2
if [[ "$*" == *"afterok:107"* && -f db/107.race ]]; then
    rm db/107.race db/107.queue
    echo '107|COMPLETED|0:0' > db/107.acct
    echo 'sbatch: error: Batch job submission failed: Job dependency problem' >&2
    exit 1
fi
job=$((600 + $(wc -l < db/submitted)))
echo "$job|PENDING" > "db/$job.queue"
if [[ -f db/wrapper_stdout ]]; then cat db/wrapper_stdout; exit 0; fi
echo "$job;cluster"
""",
    }.items():
        path=binaries/name
        path.write_text("#!/bin/bash\n"+body+"\n", newline="\n")
        path.chmod(0o755)
    for file, text in {
        "100.acct":"100.batch|FAILED|3:0\n100|COMPLETED|0:0\n",
        "101.queue":"101|RUNNING\n", "102.acct":"102|COMPLETED|1:0\n",
        "103.acct":"103|CANCELLED by 55|0:15\n", "107.queue":"107|RUNNING\n",
    }.items():
        (db/file).write_text(text,newline="\n")
    def run(command, **settings):
        env = dict(os.environ, SLURM_LOOKUP_ATTEMPTS="2",SLURM_LOOKUP_DELAY="0",**settings)
        # Let Bash put POSIX mock executables before the real scheduler.
        return subprocess.run([bash,"-c","export PATH=\"$PWD/mockbin:$PATH\"; "+command],
                              env=env,cwd=tmp_path, text=True,encoding="utf-8",capture_output=True,timeout=20)
    return run,db


@pytest.mark.parametrize("reply,expected", [
    ("sbatch: 提示：本作业申请 1 张 A800\n===== 高优先队列 =====\n  5290001  \n", "5290001"),
    ("队列 42 已就绪\n5290002;gpu-cluster\nsbatch: 提交成功\n", "5290002"),
    ("sbatch: 提示\n5290001\n5290002;gpu-cluster\n", None),
    ("sbatch: 状态代码 200，任务已提交\n=====\n", None),
])
def test_wrapper_multiline_stdout(scheduler,reply,expected):
    run,db = scheduler
    (db/"wrapper_stdout").write_text(reply,encoding="utf-8",newline="\n")
    result = run("source scripts/mri_slurm_dependencies.sh; mri_slurm_submit '' job.sh")
    assert len((db/"submitted").read_text().splitlines()) == 1
    if expected is None:
        assert result.returncode != 0 and "Check the queue" in result.stderr and not result.stdout
    else:
        assert result.returncode == 0 and result.stdout.strip() == expected


def test_submission_filters_controller_and_accounting(scheduler):
    run,db = scheduler
    (db/"101.delay").touch()  # Newly submitted job briefly missing from both sources.
    result = run("source scripts/mri_slurm_dependencies.sh; mri_slurm_submit 100:101:100 job.sh")
    assert result.returncode == 0 and result.stdout.strip()=="601", result.stderr
    assert "--dependency=afterok:101 " in (db/"submitted").read_text()
    for spec in ("102","103","12345678","100::101"):
        result = run(f"source scripts/mri_slurm_dependencies.sh; mri_slurm_submit {spec} failed.sh")
        assert result.returncode and result.stderr
    assert len((db/"submitted").read_text().splitlines()) == 1
    (db/"107.race").touch()
    result = run("source scripts/mri_slurm_dependencies.sh; mri_slurm_submit 107 race.sh")
    assert result.returncode == 0, result.stderr
    rows = (db/"submitted").read_text().splitlines()
    assert "afterok:107" in rows[-2] and "--dependency" not in rows[-1]


def test_real_submitters_use_live_dependencies(scheduler,tmp_path):
    run,db = scheduler
    ledger = tmp_path/"reports/mri/submissions/example"
    ledger.mkdir(parents=True)
    (ledger/"gate.job").write_text("100\n", newline="\n")
    result = run("bash scripts/submit_mri_haar.sh",RUN_ID="example",VARIANT="direct",TRAIN_AFTER_JOB_ID="101")
    assert result.returncode == 0 and "experiment_job=601" in result.stdout, result.stderr
    assert "--dependency=afterok:101 " in (db/"submitted").read_text()
    # SSH/admin retry during an accounting gap must not submit a duplicate gate or train job.
    (db/"601.delay").touch()
    result = run("bash scripts/submit_mri_haar.sh",RUN_ID="example",VARIANT="direct")
    assert result.returncode == 0 and len((db/"submitted").read_text().splitlines())==1
    result = run("bash scripts/submit_mri_haar.sh",RUN_ID="failed",VARIANT="direct",DEPENDENCY_JOB_ID="102")
    assert result.returncode and len((db/"submitted").read_text().splitlines())==1
    folder=tmp_path/"reports/mri/studies/check"
    folder.mkdir(parents=True)
    (folder/"backbone.tsv").write_text("example\tdirect\t2\tendpoint\t0\tidentity\t0\t42\tendpoint\n", newline="\n")
    result = run("bash scripts/submit_mri_haar_study.sh backbone",STUDY_ID="check")
    assert result.returncode==0 and "--dependency=afterok:601 " in (db/"submitted").read_text(), result.stderr
    frozen=tmp_path/"frozen"
    frozen.mkdir()
    (frozen/"matrix.json").write_text("{}")
    (frozen/"tests.tsv").write_text("testA\tfrozen/a\tout\n", newline="\n")
    conf=folder/"confirmation"
    conf.mkdir()
    (conf/"freeze.job").write_text("100\n", newline="\n")
    settings=dict(STUDY_ID="check",FREEZE_ROOT="frozen",HISTORY_ROOT="fake",STAGE1_RESULTS="fake",
                  LEGACY_RESULTS="fake",ARCHIVE_ROOT="fake",EXTERNAL_RESULTS_ROOT="fake",EXTERNAL_PREFIX="fake")
    result = run("bash scripts/submit_mri_confirmation.sh",**settings)
    assert result.returncode==0,result.stderr
    last=(db/"submitted").read_text().splitlines()[-2:]
    assert "--dependency" not in last[0] and "--dependency=afterok:603 " in last[1]
