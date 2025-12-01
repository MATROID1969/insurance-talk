#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit.components.v1 as components

_recorder = components.declare_component(
    "recorder",
    path="./recorder"
)

def record(action="none", key=None):
    return _recorder(action=action, key=key)

