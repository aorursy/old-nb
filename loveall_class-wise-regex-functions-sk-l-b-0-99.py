# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import inflect

from num2words import num2words 

import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

p = inflect.engine()





# Any results you write to the current directory are saved as output.
def cardinal(x):

    try:

        if re.match('.*[A-Za-z]+.*', x):

            return x

        x = re.sub(',', '', x, count = 10)



        if(re.match('.+\..*', x)):

            x = p.number_to_words(float(x))

        elif re.match('\..*', x): 

            x = p.number_to_words(float(x))

            x = x.replace('zero ', '', 1)

        else:

            x = p.number_to_words(int(x))

        x = x.replace('zero', 'o')    

        x = re.sub('-', ' ', x, count=10)

        x = re.sub(' and','',x, count = 10)

        return x

    except:

        return x
def digit(x): 

    try:

        x = re.sub('[^0-9]', '',x)

        result_string = ''

        for i in x:

            result_string = result_string + cardinal(i) + ' '

        result_string = result_string.strip()

        return result_string

    except:

        return(x) 
def letters(x):

    try:

        x = re.sub('[^a-zA-Z]', '', x)

        x = x.lower()

        result_string = ''

        for i in range(len(x)):

            result_string = result_string + x[i] + ' '

        return(result_string.strip())  

    except:

        return x
#Convert Roman to integers

#https://codereview.stackexchange.com/questions/5091/converting-roman-numerals-to-integers-and-vice-versa

def rom_to_int(string):



    table=[['M',1000],['CM',900],['D',500],['CD',400],['C',100],['XC',90],['L',50],['XL',40],['X',10],['IX',9],['V',5],['IV',4],['I',1]]

    returnint=0

    for pair in table:





        continueyes=True



        while continueyes:

            if len(string)>=len(pair[0]):



                if string[0:len(pair[0])]==pair[0]:

                    returnint+=pair[1]

                    string=string[len(pair[0]):]



                else: continueyes=False

            else: continueyes=False



    return returnint    

def ordinal(x):

    try:

        result_string = ''

        x = x.replace(',', '')

        x = x.replace('[\.]$', '')

        if re.match('^[0-9]+$',x):

            x = num2words(int(x), ordinal=True)

            return(x.replace('-', ' '))

        if re.match('.*V|X|I|L|D',x):

            if re.match('.*th|st|nd|rd',x):

                x = x[0:len(x)-2]

                x = rom_to_int(x)

                result_string = re.sub('-', ' ',  num2words(x, ordinal=True))

            else:

                x = rom_to_int(x)

                result_string = 'the '+ re.sub('-', ' ',  num2words(x, ordinal=True))

        else:

            x = x[0:len(x)-2]

            result_string = re.sub('-', ' ',  num2words(float(x), ordinal=True))

        return(result_string)  

    except:

        return x
def address(x):

    try:

        x = re.sub('[^0-9a-zA-Z]+', '', x)

        result_string = ''

        for i in range(0,len(x)):

            if re.match('[A-Z]|[a-z]',x[i]):

                result_string = result_string + plain(x[i]).lower() + ' '

            else:

                result_string = result_string + cardinal(x[i]) + ' '

                

        return(result_string.strip())        

    except:    

        return(x)    

    
def telephone(x):

    try:

        result_string = ''

        print(len(x))

        for i in range(0,len(x)):

            if re.match('[0-9]+', x[i]):

                result_string = result_string + cardinal(x[i]) + ' '

            else:

                result_string = result_string + 'sil '

        return result_string.strip()    

    except:    

        return(x)    



    

   