# -*- coding: utf-8 -*-
class DAGNode:

    def __init__(self, fragment, id=-1, ishead=0, istail=0):

        self.id = id
        self.fragment = fragment
        self.Source = []                
        self.weight = 1                
        self.ishead = ishead                 
        self.istail = istail                  
    def merge(self, othernode):

        self.weight += othernode.weight         
        self.ishead |= othernode.ishead              
        self.istail |= othernode.istail           
        self.Source.extend(othernode.Source)               