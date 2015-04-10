from bs4 import BeautifulSoup
import json, urllib2
import time
import re 

# pathOut = "./tids.txt"

def tidCrawler(pathOut):
	i=0
	tids = set()
	while True:
		url = "http://www.ratemyprofessors.com/search.jsp?"\
		"query=university+of+california&queryoption=HEADER"\
		"&stateselect=&country=&dept=&queryBy=teacherName&"\
		"facetSearch=true&schoolName=&offset="+str(i)+"&max=20"
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
		print i
		time.sleep(3)
	tids = list(tids)
	writer = open(pathOut, "w")
	for t in tids:
		writer.write(t + "\n")
	writer.close()


# tid = "224484"



def crawlData(tids):
	output = []
	i = 0
	for tid in tids:
		i +=1
		url = "http://www.ratemyprofessors.com/ShowRatings.jsp?tid=" + tid
		response = urllib2.urlopen(url)
		if(re.search('AddRating', response.geturl())):
			continue
		html_doc = response.read()
		soup = BeautifulSoup(html_doc)

		# basic information: fname, lname, univ, pos, dept
		top_info = soup.find_all('div',{'class':'top-info-block'})[0]

		fname = top_info.find_all('span',{'class':'pfname'})[0].getText().strip()
		lname = top_info.find_all('span',{'class':'plname'})[0].getText().strip()

		title = top_info.find_all('div',{'class':'result-title'})[0]
		univ = title.a.extract().getText()
		title = title.getText().strip()
		idx = re.search('[\r\n]', title)
		title = title[:idx.end()-1].split('in')
		pos = title[0].strip()
		dept = title[1].strip()

		# ratings: overall_quality, avg_grade, hotness, helpfulness, clarity, easiness
		rating_breakdown = soup.find_all('div',{'class':'rating-breakdown'})[0]

		ratings=rating_breakdown.find_all('div',{'class':'breakdown-header'})
		overall_quality = ratings[0].div.getText()
		avg_grade=ratings[1].div.getText()
		hotness =  ratings[2].img.get('src').split('/')[-1].split('.')[0]

		slider=rating_breakdown.find_all('div',{'class':'faux-slides'})[0].find_all('div',{'class':'rating-slider'})
		helpfulness = slider[0].find('div','rating').getText()
		clarity = slider[1].find('div','rating').getText()
		easiness = slider[2].find('div','rating').getText()

		# [overall_quality, avg_grade, hotness, helpfulness, clarity, easiness]

		# tages: taglist
		tagbox = rating_breakdown.find('div', {'class':'tag-box'})
		tagbox_choosetags = tagbox.find_all('span',{'class':'tag-box-choosetags'})
		taglist = list()
		for t in tagbox_choosetags:
			count = t.b.extract()
			count = re.sub('[^0-9]', '', count.getText())
			tagname = t.getText().strip()
			taglist.append(tuple(tagname, count))


		# reviews
		pageNum=1
		ratings = []
		while True:
			url = "http://www.ratemyprofessors.com/paginate/professors/ratings?tid=224484&page="+str(pageNum)
			response = urllib2.urlopen(url)
			json_str = response.read()
			js = json.loads(json_str)
			ratings += js['ratings']
			ramaining = js['remaining']
			pageNum += 1
			if ramaining == 0:
				break

		data = {'tid': tid, 'fname':fname, 'lname': lname, 'univ':univ, 'pos':pos, 'dept':dept, 'quality':overall_quality, 
		'avg_grade':avg_grade, 'hotness':hotness, 'helpfulness':helpfulness, 'clarity':clarity, 'easiness':easiness,
		'tags':taglist, 'ratings':ratings}
		output.append(json.dumps(data)) 
		time.sleep(5)

		if i % (len(tids)/50) and len(tids)>100:
			print("%.2f %% complete" % ( i/float(len(tids)) * 100.0))
	# end of the loop
	return output



def writeJsonOutput(path, output):
	writer = open(path, 'w')
	for line in output:
		writer.write(line+"\n")
	writer.close()

def readTid(path):
	reader = open(path, 'r')
	tids = []
	for line in reader:
		tids.append(line.strip())
	return tids

def readJson(path):
	reader = open(path, 'r')
	output = []
	for line in reader:
		js = json.loads(line.strip())
		output.append(js)
	return output

#import crawler
#reload(crawler)
#tids = crawler.readTid("./tids-umich.txt")
#output = crawler.crawlData(tids)
#crawler.writeJsonOutput("./umich.json", output)
# output = crawler.readJson("./umich.json")

