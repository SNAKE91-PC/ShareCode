'''
Created on 8 Aug 2020

@author: snake91
'''

# from housing import rightmove as rt
# 
# 
# 
# 
# url = "https://www.rightmove.co.uk/property-to-rent/find.html?searchType=RENT&locationIdentifier=REGION%5E87490&insId=1&radius=0.0&minPrice=&maxPrice=&minBedrooms=&maxBedrooms=&displayPropertyType=&maxDaysSinceAdded=&sortByPriceDescending=&_includeLetAgreed=on&primaryDisplayPropertyType=&secondaryDisplayPropertyType=&oldDisplayPropertyType=&oldPrimaryDisplayPropertyType=&letType=&letFurnishType=&houseFlatShare="
# 
# 
# x = rt.RightmoveData(url)
# 
# data = x.get_results
# 
# 
# print("")
# 

import scrapy.cmdline

def main():
    scrapy.cmdline.execute(argv=['scrapy', 'rightmove_scrapy'])

if  __name__ =='__main__':
    
    main()

    print("")