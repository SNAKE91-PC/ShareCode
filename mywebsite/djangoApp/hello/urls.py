from django.urls import path

from . import views

urlpatterns = [
    path("", views.index1, name="index"),
    path("<str:name>", views.greet0, name="greet"),
    path("<str:name>/<int:number>", views.multipleargs, name="multipleargs")
#     path("brian", views.brian, name="brian"),
#     path("david", views.david, name="david"),
]

