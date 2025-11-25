from django.urls import path
from . import views

urlpatterns = [
    # Arbres
    path('', views.principal, name='principal'),
    path('tp1/', views.tp1, name='tp1'),
    path('tree/', views.tree_view, name='tree_view'),
    path('graph/', views.graph_view, name='graph_view'),
    
    path('tp2/', views.tp2, name='tp2'),
    path('tp3/', views.tp3, name='tp3'),
    path('tp4/', views.tp4, name='tp4'),
    path('tp5/', views.tp5, name='tp5'),
    path('tp6/', views.tp6, name='tp6'),
    path('visualisation/', views.visualisation, name='visualisation'),
    path('affiche_timsort/', views.affiche_timsort, name='affiche_timsort'),
    path('simple-visualisation/', views.simple_visualisation, name='simple_visualisation'),
    
   
  
]
