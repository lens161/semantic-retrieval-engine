import subprocess
import time
import pytest


def pytest_sessionstart(session):
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.test.yml", "up", "-d"],
        check=True,
    )

    # warte auf postgres
    time.sleep(3)

def pytest_sessionfinish(session, exitstatus):
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.test.yml", "down"],
        check=True,
    )