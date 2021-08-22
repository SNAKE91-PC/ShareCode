from django.urls import path

from . import views

urlpatterns = [
#     path("", views.index0, name="index"),
#     path("", views.getStocks, name="greet"),
    path("<str:name>/<int:number>", views.getStocks, name="multipleargs"),
    path("", views.getScatter, name = "scatter")
#     path("brian", views.brian, name="brian"),
#     path("david", views.david, name="david"),
]

