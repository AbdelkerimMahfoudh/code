import pandas as pd 
import requests
from bs4 import BeautifulSoup
import csv


page = requests.get("https://www.thundafund.africa/discover")

def main (page) :

    src = page.content
    soup = BeautifulSoup(src, "lxml")
    name = soup.find("div", {'class' : 'jss40'})
    print(name)
    # def get_info (name):
    #     title = name.contents[1]
    #     print(title)
    # get_info(name)    

main(page)
 