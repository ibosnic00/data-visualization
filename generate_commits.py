#!/usr/bin/env python3
"""
Script to generate random git commits for a prettier GitHub history.
Creates 5-20 commits per day between start and end dates.
"""

import subprocess
import random
import sys
from datetime import datetime, timedelta
import os

# List of realistic commit messages
COMMIT_MESSAGES = [
    "Fix: Resolve minor bug in component rendering",
    "Refactor: Improve code organization and readability",
    "Update: Add new feature implementation",
    "Fix: Correct type definitions",
    "Update: Improve error handling",
    "Refactor: Optimize component performance",
    "Fix: Resolve linting issues",
    "Update: Enhance user experience",
    "Refactor: Clean up unused imports",
    "Fix: Correct API response handling",
    "Update: Add new utility functions",
    "Refactor: Simplify complex logic",
    "Fix: Resolve edge case in validation",
    "Update: Improve code documentation",
    "Refactor: Extract reusable components",
    "Fix: Correct date formatting",
    "Update: Add new test cases",
    "Refactor: Improve state management",
    "Fix: Resolve memory leak",
    "Update: Enhance accessibility features",
    "Refactor: Optimize bundle size",
    "Fix: Correct authentication flow",
    "Update: Add new configuration options",
    "Refactor: Improve error messages",
    "Fix: Resolve race condition",
    "Update: Improve loading states",
    "Refactor: Consolidate duplicate code",
    "Fix: Correct data validation",
    "Update: Add new icons and assets",
    "Refactor: Improve component structure",
]


def run_git_command(cmd, env=None):
    """Run a git command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            env=env,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {cmd}")
        print(f"Error: {e.stderr}")
        return None


def generate_commit(date, hour, minute, second):
    """Generate a single commit at a specific date and time."""
    # Format date for git
    date_str = date.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create environment with GIT_AUTHOR_DATE and GIT_COMMITTER_DATE
    env = os.environ.copy()
    env['GIT_AUTHOR_DATE'] = date_str
    env['GIT_COMMITTER_DATE'] = date_str
    
    # Pick a random commit message
    commit_message = random.choice(COMMIT_MESSAGES)
    
    # Create a small change to commit (modify a file or create a temp file)
    # We'll create/update a .gitkeep file in a temp directory to avoid conflicts
    temp_file = ".git_commits_temp.txt"
    
    # Write a random line to the temp file
    with open(temp_file, "a") as f:
        f.write(f"{date_str} - {random.randint(1000, 9999)}\n")
    
    # Stage the file
    run_git_command(f"git add {temp_file}", env=env)
    
    # Create the commit with the backdated timestamp
    result = run_git_command(
        f'git commit -m "{commit_message}"',
        env=env
    )
    
    if result:
        print(f"  [OK] Created commit at {date_str}: {commit_message}")
        return True
    else:
        print(f"  [FAIL] Failed to create commit at {date_str}")
        return False


def generate_commits_for_date(date):
    """Generate 5-20 random commits for a given date."""
    num_commits = random.randint(5, 20)
    print(f"\n[{date.strftime('%Y-%m-%d')}] Generating {num_commits} commits...")
    
    commits_created = 0
    
    for i in range(num_commits):
        # Random time during the day (between 9 AM and 11 PM)
        hour = random.randint(9, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        commit_datetime = date.replace(hour=hour, minute=minute, second=second)
        
        if generate_commit(commit_datetime, hour, minute, second):
            commits_created += 1
        
        # Small delay to ensure unique timestamps
        if i < num_commits - 1:
            # Add a small random delay between commits (1-30 minutes)
            delay_seconds = random.randint(60, 1800)
    
    return commits_created


def parse_date(date_str):
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format '{date_str}'. Use YYYY-MM-DD format.")
        sys.exit(1)


def main():
    """Main function to generate commits."""
    if len(sys.argv) != 3:
        print("Usage: python generate_commits.py <start_date> <end_date>")
        print("Dates should be in YYYY-MM-DD format")
        print("Example: python generate_commits.py 2024-01-01 2024-01-31")
        sys.exit(1)
    
    start_date_str = sys.argv[1]
    end_date_str = sys.argv[2]
    
    start_date = parse_date(start_date_str)
    end_date = parse_date(end_date_str)
    
    if start_date > end_date:
        print("Error: Start date must be before end date.")
        sys.exit(1)
    
    # Check if we're in a git repository
    if not run_git_command("git rev-parse --git-dir"):
        print("Error: Not in a git repository.")
        sys.exit(1)
    
    print(f"Starting commit generation...")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Create temp file if it doesn't exist
    temp_file = ".git_commits_temp.txt"
    if not os.path.exists(temp_file):
        with open(temp_file, "w") as f:
            f.write("# Temporary file for commit generation\n")
    
    total_commits = 0
    skipped_days = 0
    current_date = start_date
    
    # Generate commits for each day (20% of dates will be skipped)
    while current_date <= end_date:
        # 20% chance to skip this date (no commits)
        if random.random() < 0.2:
            print(f"\n[{current_date.strftime('%Y-%m-%d')}] Skipped (no commits)")
            skipped_days += 1
        else:
            commits_created = generate_commits_for_date(current_date)
            total_commits += commits_created
        current_date += timedelta(days=1)
    
    print(f"\n[DONE] Created {total_commits} commits total.")
    print(f"Skipped {skipped_days} days (no commits).")
    print(f"Note: The temporary file '{temp_file}' was created for commits.")
    print(f"   You can delete it if you want: git rm {temp_file} && git commit -m 'Remove temp file'")


if __name__ == "__main__":
    main()

