#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('load_ext', 'Cython')
import numpy as np




def add(x, y):
    return x + y




get_ipython().run_line_magic('timeit', 'add(1, 2)')




get_ipython().run_cell_magic('cython', '', 'cpdef int add_ints(int x, int y):\n    return x + y ')




get_ipython().run_line_magic('timeit', 'add_ints(1, 2)')




get_ipython().run_cell_magic('cython', '', 'cpdef float add_floats(float x, float y):\n    return x + y ')




get_ipython().run_line_magic('timeit', 'add_floats(1.0, 2.0)')




get_ipython().run_line_magic('timeit', 'add_floats(1, 2)')

