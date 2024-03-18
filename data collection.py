# import pandas as pd 
import requests
from bs4 import BeautifulSoup
import csv


page = requests.get("https://web.archive.org/web/20220302234213/http://www.rimnow.com/")

def main (page) :

    src = page.content
    # soup = BeautifulSoup(src, "lxml")
    print(soup)

main(page)
 