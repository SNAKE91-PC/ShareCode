


from django.http import HttpResponse
from django.shortcuts import render

def index0(request):
    return HttpResponse("Hello, world!")

def index1(request):
    return render(request, "dashboard/index.html")

# def brian(request):
#     return HttpResponse("Hello, Brian0!")
# 
# def david(request):
#     return HttpResponse("Hello, David!")

def greet0(request, name):
    return HttpResponse(f"Hello1, {name.capitalize()}!")

def greet1(request, name):
    return render(request, "dashboard/index.html", {
        "name": name.capitalize()
    })
    
    
def multipleargs(request, name, number):
    return HttpResponse(f"Sono qui, {name}, the number is {number}")



def getStocks(request):
    return render(request, "dashboard/index.html")


def getScatter(request):
    return render(request, "dashboard/scatter.html")