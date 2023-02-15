# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:34:45 2022

@author: Tomás Wierzba
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import numpy_financial as npf
import math
import altair as alt



#from IPython.display import Image
#import base64, io, IPython
from PIL import Image 
image = Image.open('HG_logo_white text and box_hori.png')
image2 = Image.open('HG_Yellow_hori.png')


st.image(image, caption=None)

#new_title0 = '<p style="font-size:45px;font-weight:700;color:black;text-align:center;">NEXP2X Business-Case Tool</p>' 
#st.write(new_title0, unsafe_allow_html=True)
st.write(""" # E-Ammonia Business-Case Tool """)

st.write(""" Created in the ClusterSoutH2 Project - Funded by Energy Cluster Denmark """)




#Explain assumptions here

electrolyser_nom_cap = 1000 #kW

st.sidebar.image(image2)

new_title1 = '<p style="font-size:25px;font-weight:600;color:#f0f2f6">Key variables</p>'
st.sidebar.write(new_title1, unsafe_allow_html=True)
new_title3 = '<p style="font-size:15px;font-weight:500;color:#f0f2f6">This tool assumes a 1 MW electrolyzer. Main variables for this business case-study can be changed in the left pane and their initial values repesent AEC technology.</p>'
st.sidebar.write(new_title3, unsafe_allow_html=True)

new_title2 = '<p style="font-size:20px;font-weight:600;color:#f0f2f6">Market prices</p>'
st.sidebar.markdown(new_title2, unsafe_allow_html=True)
#Decide future Hydrogen price
Ammonia_price = st.sidebar.slider('Average Ammonia sales price in €/ton:',0, 2000,750,50)


#Decide average electricity spot price
Electricity_spot_MWh = st.sidebar.slider('Average electricity spot price in €/MWh: ', 0, 200, 50,10)
Electricity_spot = Electricity_spot_MWh/1000

new_title4 = '<p style="font-size:20px;font-weight:600;color:#f0f2f6">Electrolyzer</p>'
st.sidebar.markdown(new_title4, unsafe_allow_html=True)

#Decide electrolyzer technology
elec_technology = st.sidebar.selectbox(
    'Select electrolyzer technology, custom your own or go for a PPA:',
    ('AEC','SOEC', 'Custom', 'Get green H\u2082 through PPA'))

if elec_technology=='Custom':
    full_load_hours= st.sidebar.slider('Electrolyzer full-load hours of operation in a year: ', 0, 8760, 7500,100)
    H2_electrolyser_input_1000 = st.sidebar.slider('Power-to-Hydrogen production ratio in kg/MWh: ', 10, 30, 20,1)
    electrolyser_specific_invest= st.sidebar.slider('Electrolyzer capital investment in €/kW: ', 0, 5000, 1000,250)
    technical_lifetime_stacks= st.sidebar.slider('Stacks lifetime in full-load hours of operation: ', 0, 100000,100000 ,5000)
    electrolyser_OPEX_percentage2= st.sidebar.slider('O&M yearly in % CAPEX (excluding Stack replacement cost): ', 0, 20, 5,1)
    electrolyser_STACK_replacement_100 = st.sidebar.slider('Stack replacement cost as % of CAPEX: ', 0, 50,25,1)
elif elec_technology == 'SOEC':
    full_load_hours= st.sidebar.slider('Electrolyzer full-load hours of operation in a year: ', 0, 8760, 7500,100)
    H2_electrolyser_input_1000=23.3
    technical_lifetime_stacks=20000
    electrolyser_specific_invest=3000
    electrolyser_OPEX_percentage2=5
    electrolyser_STACK_replacement_100=20
elif elec_technology == 'AEC':
    full_load_hours= st.sidebar.slider('Electrolyzer full-load hours of operation in a year: ', 0, 8760, 7500,100)
    H2_electrolyser_input_1000=20
    technical_lifetime_stacks=100000
    electrolyser_specific_invest=750
    electrolyser_OPEX_percentage2=5
    electrolyser_STACK_replacement_100=30
elif elec_technology == 'Get green H\u2082 through PPA':
    Hydrogen_cost = st.sidebar.slider('H\u2082 cost in €/kg: ',0, 15,3,1)  
    Hydrogen_amount_purchased = st.sidebar.slider('H\u2082 purchased in ton/year: ',0, 85,10,1)
if elec_technology !='Get green H\u2082 through PPA':
    H2_electrolyser_input = H2_electrolyser_input_1000/1000
    electrolyser_OPEX_percentage = electrolyser_OPEX_percentage2/100
    electrolyser_STACK_replacement = electrolyser_STACK_replacement_100/100

new_title4b = '<p style="font-size:20px;font-weight:600;color:#f0f2f6">Ammonia Synthesis</p>'
st.sidebar.markdown(new_title4b, unsafe_allow_html=True)

Ammonia_H2_conversion = st.sidebar.slider('Conversion factor as % of H\u2082 converted: ', 50, 100, 90,1)
Ammonia_specific_invest = st.sidebar.slider('Ammonia capital investment in €/(kg/h) ammonia: ', 0, 5000, 3000,250)
Ammonia_OPEX_percentage = st.sidebar.slider('O&M yearly in % CAPEX: ', 0, 20, 2,1)
    
new_title5 = '<p style="font-size:20px;font-weight:600;color:#f0f2f6">Financial</p>'
st.sidebar.markdown(new_title5, unsafe_allow_html=True)

#Decide project lifetime
lifetime = st.sidebar.slider('Project lifetime in years:', 0, 30,25,1)

#Decide discount rate
discountRate_100 = st.sidebar.slider('Desired discount rate in %:', 0, 50, 5,1)
discountRate = discountRate_100/100


#------------------------------------Income-----------------------------------------------------------------------------------------
Hydrogen_input_yearly = np.zeros(lifetime +1)   #in kg
for t in range(1,lifetime+1):
    if elec_technology=='Get green H\u2082 through PPA':
        Hydrogen_input_yearly[t] = Hydrogen_amount_purchased*1000
    else:
        Hydrogen_input_yearly[t] = H2_electrolyser_input * full_load_hours * electrolyser_nom_cap
ammonia_prod_yearly = np.zeros(lifetime+1)   #in tons
for t in range(1,lifetime+1):  
    ammonia_prod_yearly[t] = Hydrogen_input_yearly[t]*Ammonia_H2_conversion/100*34/6/1000  #34/6 is the reaction stechiometric factor
ammonia_income_yearly = np.zeros(lifetime +1)
for t in range(1,lifetime+1):
    ammonia_income_yearly[t] = Ammonia_price * ammonia_prod_yearly[t] #€/year
#Since ammonia synthesis facilities are designed to work with a maximum flow - and the correwsponding capital investment is based on this. Due to the fact that the maximum hydrogen production in one hour is 20kg (for a 1 MW AEC, depends on technology), then the maximum production capacity will be given by:
if elec_technology=='Get green H\u2082 through PPA':  
    max_ammonia_prod_flow = Hydrogen_amount_purchased*1000/8760*34/6*Ammonia_H2_conversion/100 #assuming constant h2 input
else:
    max_ammonia_prod_flow = H2_electrolyser_input*1000*34/6*Ammonia_H2_conversion/100
#-----------------------------------OPEX & CAPEX---------------------------------------------------------------------------------------

if elec_technology=='Get green H\u2082 through PPA':    
    OPEX_electrolyser_yearly = Hydrogen_cost*Hydrogen_amount_purchased*1000 #this is actually the cost of purchasing the Hydrogen through the PPA
    Electricity_cost_yearly=0
    CAPEX_electrolyser=0
else:   
    OPEX_electrolyser_yearly = electrolyser_OPEX_percentage * electrolyser_specific_invest * electrolyser_nom_cap #€/year this value must be changed when changing electrolyser specific investment
    Electricity_cost_yearly  = Electricity_spot * full_load_hours * electrolyser_nom_cap
    CAPEX_electrolyser      = electrolyser_nom_cap * electrolyser_specific_invest #€ this value must be changed when changing electrolyser specific investment
    
OPEX_ammonia_synthesis = Ammonia_OPEX_percentage/100*Ammonia_specific_invest*ammonia_prod_yearly[1]
OPEX_yearly              = OPEX_electrolyser_yearly + Electricity_cost_yearly + OPEX_ammonia_synthesis #€/year this is the yearly OPEX
CAPEX_ammonia = Ammonia_specific_invest*max_ammonia_prod_flow
CAPEX                   = CAPEX_electrolyser + CAPEX_ammonia
#Replacement cost for stacks when technical lifetime is achieved included in cash flow


#------------------------------------CashFlow-----------------------------------------------------------------------------------------
cf = np.zeros(lifetime+1) #cashflow in M€
for t in range (1, len(cf)):
    cf[t] = (- OPEX_yearly + ammonia_income_yearly[t])/1e+6
cf[0] = -CAPEX/1e+6
if elec_technology !='Get green H\u2082 through PPA':
    electrolyser_exact_replacement_period = technical_lifetime_stacks/full_load_hours #years
    years_of_stack_replacement = np.zeros(lifetime+1)
    years_of_stack_replacement[0]=0
    for i in range(1, lifetime+1):
        if math.ceil(electrolyser_exact_replacement_period*i)<=lifetime:
            years_of_stack_replacement[math.ceil(electrolyser_exact_replacement_period*i)]=1
    for t in range (1,len(cf)):
        cf[t] = -CAPEX_electrolyser * electrolyser_STACK_replacement * years_of_stack_replacement[t] /1e+6 + (- OPEX_yearly + ammonia_income_yearly[t])/1e+6

#------------------------------------NPV--------------------------------------------------------------------------------------
discountRate2 = round(discountRate*100,1)
npv             = npf.npv(discountRate, cf)
npv2 = round(npv,1)
NPV = np.zeros(len(cf))
for i in range(0,len(cf)):
    NPV[i] = npf.npv(discountRate, cf[0:(i+1)])
for i in range(1,len(cf)):
    if NPV[i]>=0:
        if NPV[i-1]<=0:
            a1010=i

if all(e <= 0 for e in NPV):
    a101="N/A"
    #st.write('Project is not profitable in 27 years')
else:
    a101 = "%s years" % (a1010)
    #st.write('Payback time: %s years' % (a1010))

#st.write('Net present value: %s M€ (%s %% discount rate)' % (npv2,discountRate2))
#------------------------------------IRR---------------------------------------------------------------------------------------------
IRR = npf.irr(cf)
b = np.where(np.isnan(IRR), -1000, IRR)
if b==-1000:
    IRR2="N/A"
else:
   IRR2 = "%s %%" % (round(IRR*100))

#st.write('IRR: %s %%' % (IRR2))
#Hydrogen Price independent
#------------------------------------LCoH-----------------------------------------------------------------------------------
Expenses = np.zeros(lifetime +1) #expenses plus electricity income in €/year
for t in range(1,len(cf)):
    Expenses[t] = -cf[t]*1e+6 + (ammonia_income_yearly[t])
Expenses[0] = -cf[0]*1e+6
LCoH = npf.npv(discountRate,Expenses)/npf.npv(discountRate, ammonia_prod_yearly)
LCoH2 = round(LCoH,1)
#st.write('Levelized Cost of Hydrogen: %s €/kg (%s %% discount rate)' % (LCoH2,discountRate2))

#------------------------------------LCoH per expense-----------------------------------------------------------------
OPEX_electrolyser_yearly_v = np.zeros(lifetime+1)
OPEX_electrolyser_yearly_v[0] = 0
for i in range(1,lifetime+1):
    OPEX_electrolyser_yearly_v[i] = OPEX_electrolyser_yearly
LCoH_opex_electrolyser = npf.npv(discountRate,OPEX_electrolyser_yearly_v)/npf.npv(discountRate, ammonia_prod_yearly)
LCoH_opex_electrolyser2 = round(LCoH_opex_electrolyser,1)

OPEX_ammonia_yearly_v = np.zeros(lifetime+1)
OPEX_ammonia_yearly_v[0] = 0
for i in range(1,lifetime+1):
    OPEX_ammonia_yearly_v[i] = OPEX_ammonia_synthesis
LCoH_opex_ammonia = npf.npv(discountRate,OPEX_ammonia_yearly_v)/npf.npv(discountRate, ammonia_prod_yearly)
LCoH_opex_ammonia2 = round(LCoH_opex_ammonia,1)

Electricity_cost_yearly_v = np.zeros(lifetime+1)
Electricity_cost_yearly_v[0] = 0
for i in range(1,lifetime+1):
    Electricity_cost_yearly_v[i] = Electricity_cost_yearly
LCoH_electricity_cost = npf.npv(discountRate,Electricity_cost_yearly_v)/npf.npv(discountRate, ammonia_prod_yearly)
LCoH_electricity_cost2 = round(LCoH_electricity_cost,1)


CAPEX_elec_v = np.zeros(lifetime+1)
CAPEX_elec_v[0] = CAPEX_electrolyser
LCoH_capex_elec = npf.npv(discountRate,CAPEX_elec_v)/npf.npv(discountRate, ammonia_prod_yearly)
LCoH_capex_elec_2 = round(LCoH_capex_elec,1)


CAPEX_am_v = np.zeros(lifetime+1)
CAPEX_am_v[0] = CAPEX_ammonia
LCoH_capex_am = npf.npv(discountRate,CAPEX_am_v)/npf.npv(discountRate, ammonia_prod_yearly)
LCoH_capex_am_2 = round(LCoH_capex_am,1)

if elec_technology !='Get green H\u2082 through PPA':
    stack_replacement_cost_v = np.zeros(lifetime+1)
    stack_replacement_cost_v[0] = 0
    for t in range(1,len(cf)):
        stack_replacement_cost_v[t] = CAPEX_electrolyser * electrolyser_STACK_replacement * years_of_stack_replacement[t]
    LCoH_stack_rep_cost = npf.npv(discountRate,stack_replacement_cost_v)/npf.npv(discountRate, ammonia_prod_yearly)
    LCoH_stack_rep_cost2 = round(LCoH_stack_rep_cost,1)
else:
    LCoH_stack_rep_cost2=0
if elec_technology !='Get green H\u2082 through PPA':
    data = {
    'Electricity':LCoH_electricity_cost2,'CAPEX Ammonia':LCoH_capex_am_2,'CAPEX Hydrogen':LCoH_capex_elec_2,'Stack Replacement':LCoH_stack_rep_cost2, 'O&M Electrolyzer':LCoH_opex_electrolyser2
    , 'OPEX Ammonia Syn.':LCoH_opex_ammonia2}
    a20 = max(data, key=data.get)
    per_main_costdriver = round(data[a20] / LCoH * 100 )
else:
    data = {
    'Electricity':LCoH_electricity_cost2,'CAPEX Ammonia':LCoH_capex_am_2,'CAPEX Hydrogen':LCoH_capex_elec_2,'Stack Replacement':LCoH_stack_rep_cost2, 'Green Hydrogen cost PPA':LCoH_opex_electrolyser2
    , 'OPEX Ammonia Syn.':LCoH_opex_ammonia2}
    a20 = max(data, key=data.get)
    per_main_costdriver = round(data[a20] / LCoH * 100 )
#------------------------------------Show results-----------------------------------------------------------------
#new_title7 = '<p style="font-size:45px;font-weight:700;color:black;text-align:center;">Results</p>'
#st.write(new_title7, unsafe_allow_html=True)
st.write(""" # Results """)
col1, col2 , col3= st.columns(3)
col1.metric("Payback time", '%s' % (a101))
col3.metric("IRR", "%s" % (IRR2))
col2.metric("NPV", "%s M€/MW"  % (npv2))
col5, col6,col7= st.columns(3)
col5.metric("LCoH", "%s €/ton" % (LCoH2))
(col6+col7).metric("Cost-driver","%s (%s %% of cost)" % (a20, per_main_costdriver))
#st.write("The main cost-driver for the Levelized Cost of Hydrogen is found to be %s, accounting for %s %% of the cost." % (a20, per_main_costdriver))

st.write(" # Levelised cost contributions for Ammonia")

if elec_technology !='Get green H\u2082 through PPA': 
    source = pd.DataFrame({"Values": [LCoH_electricity_cost2,LCoH_capex_am_2,LCoH_capex_elec_2,LCoH_opex_ammonia2,LCoH_stack_rep_cost2, LCoH_opex_electrolyser2],"Cost contribution": ['Electricity: %s €/ton' % (LCoH_electricity_cost2),'CAPEX Ammonia Synthesis: %s €/ton' % (LCoH_capex_am_2),'CAPEX Electrolyzer: %s €/ton' % (LCoH_capex_elec_2),'O&M Ammonia Synthesis: %s €/ton' % (LCoH_opex_ammonia2),'Stack Replacement: %s €/ton' % (LCoH_stack_rep_cost2),'O&M Electrolyzer: %s €/ton' % (LCoH_opex_electrolyser2)],"labels":["%s €/ton" % (LCoH_electricity_cost2),"%s €/ton" % (LCoH_capex_am_2),"%s €/ton" % (LCoH_capex_elec_2),"%s €/ton" % (LCoH_stack_rep_cost2),"%s €/ton" % (LCoH_opex_electrolyser2),"%s €/ton" % (LCoH_opex_ammonia2)]})
    domain = ['Electricity: %s €/ton' % (LCoH_electricity_cost2),'CAPEX Ammonia Synthesis: %s €/ton' % (LCoH_capex_am_2),'CAPEX Electrolyzer: %s €/ton' % (LCoH_capex_elec_2),'O&M Ammonia Synthesis: %s €/ton' % (LCoH_opex_ammonia2),'Stack Replacement: %s €/ton' % (LCoH_stack_rep_cost2),'O&M Electrolyzer: %s €/ton' % (LCoH_opex_electrolyser2)]
    range_ = ['#088da5', 'grey', '#f0f2f6', '#ffe300','blue','red']
    base = alt.Chart(source).encode(
        theta=alt.Theta("Values:Q", stack=True), color=alt.Color('Cost contribution:N', scale=alt.Scale(domain=domain, range=range_),legend=alt.Legend(clipHeight=50)),
        radius=alt.Radius("Values:Q", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
    )

    c1 = base.mark_arc(innerRadius=20)

    #c2 = base.mark_text(radiusOffset=45).encode(text="labels:N")


    rp=(c1).configure_text(fontSize=25,fontWeight=600).configure_legend(titleFontSize=22, titleFontWeight=600,labelFontSize= 20,labelFontWeight=600,labelLimit= 0)#.configure(autosize="fit")

    st.altair_chart(rp, use_container_width=True)
else: 
    source = pd.DataFrame({"Values": [LCoH_capex_am_2,LCoH_opex_ammonia2, LCoH_opex_electrolyser2],"Cost contribution": ['CAPEX Ammonia Synthesis: %s €/ton' % (LCoH_capex_am_2),'O&M Ammonia Synthesis: %s €/ton' % (LCoH_opex_ammonia2),'Green Hydrogen cost PPA: %s €/ton' % (LCoH_opex_electrolyser2)],"labels":["%s €/ton" % (LCoH_capex_am_2),"%s €/ton" % (LCoH_opex_electrolyser2),"%s €/ton" % (LCoH_opex_ammonia2)]})
    domain = ['CAPEX Ammonia Synthesis: %s €/ton' % (LCoH_capex_am_2),'O&M Ammonia Synthesis: %s €/ton' % (LCoH_opex_ammonia2),'Green Hydrogen cost PPA: %s €/ton' % (LCoH_opex_electrolyser2)]
    range_ = [ 'grey','#ffe300','#a2c11c']
    base = alt.Chart(source).encode(
        theta=alt.Theta("Values:Q", stack=True), color=alt.Color('Cost contribution:N', scale=alt.Scale(domain=domain, range=range_),legend=alt.Legend(clipHeight=50)),
        radius=alt.Radius("Values:Q", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
    )

    c1 = base.mark_arc(innerRadius=20)

    #c2 = base.mark_text(radiusOffset=45).encode(text="labels:N")


    rp=(c1).configure_text(fontSize=25,fontWeight=600).configure_legend(titleFontSize=22, titleFontWeight=600,labelFontSize= 20,labelFontWeight=600,labelLimit= 0)#.configure(autosize="fit")

    st.altair_chart(rp, use_container_width=True)    
brush2 = alt.selection_interval()
st.write(" # Cash flow plots")
year=np.linspace(0, lifetime,lifetime+1)
chart_data2 = pd.DataFrame({'Year':year,'Non-discounted Cash Flows in Million €':cf})
d = alt.Chart(chart_data2).mark_bar().encode(
     x='Year:O',y='Non-discounted Cash Flows in Million €:Q',color=alt.value('#ffe300'))

xposim2 = round(lifetime/2)
yposim2 = (cf[0] + cf[len(cf) - 1])/2

source2 = pd.DataFrame.from_records([
      {"Year": xposim2, "Non-discounted Cash Flows in Million €": yposim2, "imga2": "https://raw.githubusercontent.com/tomaswierzba/P2X/main/HG_Yellow_hori.png"}
])

img2 = alt.Chart(source2).mark_image(opacity=0.5,
    width=300,
    height=100
).encode(
    x='Year:O',
    y='Non-discounted Cash Flows in Million €:Q',
    url='imga2'
)

k=(d+img2).interactive().properties(    #color=alt.condition(brush2, alt.value('#ffe300'), alt.value('lightgray'))
    title='Non-discounted Cash Flows',width= 600, height= 400
).configure_title(
    fontSize=25,
    fontWeight=900,
    anchor='middle',
    color='#f0f2f6'
).configure_axis(titleColor='#f0f2f6',labelColor='#f0f2f6',labelAngle=0,labelFontSize=15,titleFontSize=15, gridColor='black') 

st.altair_chart(k, use_container_width=True) 

brush = alt.selection_interval()
chart_data3 = pd.DataFrame({'Year':year,"Acc Disc Cash Flows in Million €":NPV})

c = alt.Chart(chart_data3).mark_bar().encode(
     x='Year:O',y="Acc Disc Cash Flows in Million €", color=alt.value('#ffe300') )


xposim = round(lifetime/2)
yposim = (NPV[0] + NPV[len(cf) - 1])/2

source = pd.DataFrame.from_records([
      {"Year": xposim, "Acc Disc Cash Flows in Million €": yposim, "imga": "https://raw.githubusercontent.com/tomaswierzba/P2X/main/HG_Yellow_hori.png"}
])

img = alt.Chart(source).mark_image(opacity=0.5,
    width=300,
    height=100
).encode(
    x='Year:O',
    y='Acc Disc Cash Flows in Million €',
    url='imga'
)

if all(e <= 0 for e in NPV):
    g=(c+img).interactive().properties(
        title='Accumulated Discounted Cash Flows',width= 600, height= 400).configure_title(fontSize=25,fontWeight=900,anchor='middle',color='#f0f2f6').configure_axis(titleColor='#f0f2f6',labelColor='#f0f2f6',labelAngle=0,labelFontSize=15,titleFontSize=15, gridColor='black').configure_line(fontStyle='dash', fontWeight=900).configure_text(fontSize=15,fontWeight='bold')
    st.altair_chart(g, use_container_width=True) 
else:
    for i in range(0,len(cf)):
        year[i]=a1010

    label=['']*len(cf)
    label[1]='Payback Time'
    chart_data4 = pd.DataFrame({'Year':year,"Acc Disc Cash Flows in Million €":NPV, "Label":label})    

    line = alt.Chart(chart_data4).mark_rule(color='red').encode( x='Year:O',y="Acc Disc Cash Flows in Million €")

    text = line.mark_text(
        align='right',
        baseline='middle',
        dx=-10
    , color= 'red').encode(
        text='Label'
    )

    g=(c+line+text+img).interactive().properties(
        title='Accumulated Discounted Cash Flows',width= 600, height= 400).configure_title(fontSize=25,fontWeight=900,anchor='middle',color='#f0f2f6').configure_axis(titleColor='#f0f2f6',labelColor='#f0f2f6',labelAngle=0,labelFontSize=15,titleFontSize=15, gridColor='black').configure_line(fontStyle='dash', fontWeight=900).configure_text(fontSize=15,fontWeight='bold') #.configure_image(opacity=0.5,width=50,height=50)
    st.altair_chart(g, use_container_width=True) 
    

st.write(" #  What's your next action towards 100% renewables?")
new_title100 = '<p style="font-size:20px;font-weight:600;color:#f0f2f6;"><span> Let&#8217s create more value together, send us an e-mail to </span><span id="name"><a href = "mailto: info@hybridgreentech.com" style="color:#ffe300">info@hybridgreentech.com</a></span></p>'
st.write(new_title100, unsafe_allow_html=True)
#st.write("Let's create more value together, send us an e-mail to info@hybridgreentech.com")
