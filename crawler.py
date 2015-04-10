from bs4 import BeautifulSoup
import json, urllib2


i=1
cnt =1
tids = set()
while True:
	url = "http://www.ratemyprofessors.com/search.jsp"\
		"?query=university+of+michigan&queryoption=HEADER"\
		"&stateselect=&country=&dept=&queryBy=teacherName&"\
		"facetSearch=true&schoolName=university+of+michigan&offset="+str(i)+"&max=20"
	response = urllib2.urlopen(url)
	html_doc = response.read()
	soup = BeautifulSoup(html_doc)
	rows=soup.find_all('li',{'class':'listing PROFESSOR'})
	if len(rows) == 0:
		break
	for row in rows:
		link = row.a.get('href')
		idx = re.search('tid=', link).end()
		tid = link[idx:]
		tids.add(tid)
	i+=20
	cnt +=1
	time.sleep(5)
	if cnt==5:
		break


tids = list(tids)
writer = open("tids.txt", "w")
for t in tids:
	writer.writer(t + "\n")

writer.close()


# reviews
http://www.ratemyprofessors.com/paginate/professors/ratings?tid=63035&page=3



