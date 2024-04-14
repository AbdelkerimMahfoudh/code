import pandas as pd 
import requests
from bs4 import BeautifulSoup
import csv


page = requests.get("https://www.kickstarter.com/discover/advanced?woe_id=24865670&sort=popularity&seed=2852995&page=12&next_page_cursor=")

def main (page) :

    src = page.content
    soup = BeautifulSoup(src, "lxml")
    

    name = soup.find_all("span", {'class': 'card-title'})
    print(name)

main(page)
 