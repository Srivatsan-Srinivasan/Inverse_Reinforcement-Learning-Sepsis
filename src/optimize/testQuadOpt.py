# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:44:04 2017

@author: SrivatsanPC
"""

import quadOpt as qo
import random
import pdb
#pdb.set_trace()
q = qo.quadOpt()

#define mu as a two dimensional vector.

target_mu = [3,4]
candiate_mu_final = [3.1, 4.1]
test_size = 30 

for i in range(test_size-1):
    x = round(random.uniform(3.5,5),2)
    y = round(random.uniform(4.5,5),2)
    
    print(q.optOutput(target_mu,[x,y]))
    