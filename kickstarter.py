import requests
from bs4 import BeautifulSoup


url = "https://www.kickstarter.com/discover/advanced?woe_id=24865670&sort=magic&seed=2852995&next_page_cursor=&page=1"
result = requests.get(url).text
doc = BeautifulSoup(result,"html.parser")

divs = doc.div
print(divs)

   


 