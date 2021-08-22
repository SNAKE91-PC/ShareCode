# -*- coding: utf-8 -*-
import scrapy
import json
import pandas as pd
import pprint
import numpy as np
from flatten_json import flatten
from scrapy.linkextractors import LinkExtractor

class ToScrapeCSSSpider(scrapy.Spider):
    name = "toscrape-css"
    start_urls = [
        #'http://quotes.toscrape.com/',
        'https://www.rightmove.co.uk/property-to-rent/find.html?locationIdentifier=REGION%5E87401'

    ]


    def parse(self, response):

        options = '&propertyTypes=&includeLetAgreed=false&mustHave=&dontShow=retirement%2ChouseShare%2Cstudent&furnishTypes=&keywords='

        # quote = response.css("a.propertyCard-link")

        print(response.xpath('//span[@class="pagination-pageInfo"]'))

        jsonDict = list(filter(lambda x: "window.jsonModel" in str(x), response.xpath('//script')))[0].extract()
        jsonDict = jsonDict.split("window.jsonModel = ")[1].split("</script")[0]

        jsonDict = json.loads(jsonDict)

        jsonList = jsonDict["properties"]

        jsonCleanList = [flatten(Dict) for Dict in jsonList]

        data = pd.DataFrame(jsonCleanList)

        # response.xpath('//button[@class = "pagination-button pagination-direction pagination-direction--next"')

        for i in range(1, 42):
            var = "https://www.rightmove.co.uk/property-to-rent/find.html?locationIdentifier=REGION%5E87401"

            var += "&index=" + str(i * 24)  # 25 per page

            var += options

            yield scrapy.Request(var)


        # next_page_url = response.css("li.next > a::attr(href)").extract_first()
        # if next_page_url is not None:
        #     yield scrapy.Request(response.urljoin(next_page_url))

