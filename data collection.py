import pandas as pd 
import requests
from bs4 import BeautifulSoup
import csv


page = requests.get("https://www.thundafund.africa/discover")

def main (page) :

    src = page.content
    soup = BeautifulSoup(src, "lxml")
    projects = soup.find_all(class_="MuiGrid-root")
    def get_prj_info(projects):
        project_name = projects.contents
        print(project_name)
   
    get_prj_info(projects[0])
main(page)
 