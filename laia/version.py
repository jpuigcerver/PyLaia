from __future__ import absolute_import

import os
import subprocess


def git_commit():
    """
    Returns the hash of the current Git commit, or None if the package is not under Git.
    """
    stderr = open(os.devnull, 'w')
    cwd = os.getcwd()
    commit = None
    try:
        os.chdir(os.path.dirname(__file__))
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=stderr)
        commit = commit.strip()
    finally:
        os.chdir(cwd)
        stderr.close()
        return commit


def git_branch():
    """
    Returns the name of the current Git branch, or None if the package is not under Git.
    """
    stderr = open(os.devnull, 'w')
    cwd = os.getcwd()
    branch = None
    try:
        os.chdir(os.path.dirname(__file__))
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=stderr)
        branch = branch.strip()
    finally:
        os.chdir(cwd)
        stderr.close()
        return branch


def git_describe():
    """Describes the current Git commit using the most recent tag."""
    stderr = open(os.devnull, 'w')
    cwd = os.getcwd()
    describe = None
    try:
        os.chdir(os.path.dirname(__file__))
        describe = subprocess.check_output(
            ['git', 'describe', '--tags', '--dirty', '--always'], stderr=stderr)
        describe = describe.strip()
    finally:
        os.chdir(cwd)
        stderr.close()
        return describe
