import requests
from bs4 import BeautifulSoup
import csv

url = "https://www.thundafund.africa/discover"
result = requests.get(url).text
doc = BeautifulSoup(result,"html.parser")

divs = doc.div
min = divs.contents
print(list(min[1].descendants))
   

# def main (page) :

#     src = page.content
#     soup = BeautifulSoup(src, "lxml")
#     projects = soup.find_all(class_="MuiGrid-root")
#     def get_prj_info(projects):
#         project_name = projects.contents
#         print(project_name)
   
#     get_prj_info(projects[0])
# main(page)
 