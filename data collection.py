import pandas as pd 
import requests
from bs4 import BeautifulSoup
import csv

date = "Please enter a date in the following format MM/DD/YYYY : "
page = requests.get("https://www.yallakora.com/Match-Center/?date=12/13/2023")