import lxml.html
import urllib2
import os
import csv

url = str('http://wogma.com/genres/')
page=urllib2.urlopen(url)
dom =  lxml.html.fromstring(page.read())
link = dom.xpath('//tr[@class="odd" or @class ="even"]/td/a/@href')

print(link)

for i in link:
    url = str('http://wogma.com'+i)
    page=urllib2.urlopen(url)
    dom =  lxml.html.fromstring(page.read())
    movielink = dom.xpath('//td[@class="listing_synopsis"]/div[@class="button related_pages review "]/a/@href')
    
    
    savepath = 'C:/Users/Ramkumar.Ramkumar-Lappy/Desktop/Topic Modeling Using Python/Data'
    genre = "".join(i)
    name_of_file= genre.split('/')[2]
    completeName = os.path.join(savepath, name_of_file+".csv")  
    csvfile = open(completeName, 'wb')
    writer = csv.writer(csvfile)

    
    for j in movielink:
        url = str('http://wogma.com'+j)
        page=urllib2.urlopen(url)
        dom =  lxml.html.fromstring(page.read())
        review = dom.xpath('//div[@class="review large-first-letter"]/p/text()')
        review = ''.join(review)
        review = review.encode('utf-8')
        quickreview = dom.xpath('//span[@class="quick_review"]/p/text()')
        quickreview = ''.join(quickreview)
        quickreview = quickreview.encode('utf-8')
        mergelist=quickreview+review
        writer.writerow([mergelist])

    csvfile.close()