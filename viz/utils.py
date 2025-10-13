import os

def get_week_path(year: int, week: int):
  return os.path.join(os.path.dirname(__file__), "..", str(year), f"week-{week}")
