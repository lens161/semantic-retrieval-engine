import subprocess
import time


def pytest_sessionstart(session):
    subprocess.run(
        [
            "docker", "compose",
            "-f", "docker-compose.test.yml",
            "-p", "semantic-test",   # 👈 WICHTIG
            "up", "-d"
        ],
        check=True,
    )
    # warte auf postgres
    time.sleep(3)

def pytest_sessionfinish(session, exitstatus):
    subprocess.run(
        [
            "docker", "compose",
            "-f", "docker-compose.test.yml",
            "-p", "semantic-test",
            "down", "-v"
        ],
        check=True,
    )