# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 18:35:10 2017

@author: Rahul.kumar
"""

import web
#import sys,os
import json
import sentiment_model as model

urls = ('/sentiment', 'Sentiment')
app = web.application(urls, globals())

class Sentiment:
    def GET(self):
        web.header('Content-Type', 'application/json')
        user_data = web.input()
        comment= user_data.message
#        commentId= user_data.commentId
#        senderid= user_data.senderid
#        senderName= user_data.senderName
            
        sentiment = model.engine(query = comment , senderid='Rahul', senderName='Rahul')
            
        return json.dumps(sentiment )



if __name__ == "__main__":
    app.run()
