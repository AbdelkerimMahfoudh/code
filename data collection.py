import requests
from bs4 import BeautifulSoup
import csv

url = "https://www.thundafund.africa/discover"
result = requests.get(url).text
doc = BeautifulSoup(result,"html.parser")

divs = doc.div
src = divs.contents
for div in src[:4]:
    name = div.contents[1:3]
    print(name)

   

# def main (page) :

#     src = page.content
#     soup = BeautifulSoup(src, "lxml")
#     projects = soup.find_all(class_="MuiGrid-root")
#     def get_prj_info(projects):
#         project_name = projects.contents
#         print(project_name)
   
#     get_prj_info(projects[0])
# main(page)
 