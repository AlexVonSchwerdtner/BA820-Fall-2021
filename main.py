# imports
import pandas as pd

# sql
SQL = "select * from `questrom.datasets.mtcars`"
PROJECT = "ba820-avs"

df = pd.read_gbq(SQL, PROJECT)